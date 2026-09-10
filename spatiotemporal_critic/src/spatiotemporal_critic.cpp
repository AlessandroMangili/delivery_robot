#include "spatiotemporal_critic/spatiotemporal_critic.hpp"

#include "geometry_msgs/msg/point.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

#include <xtensor/xmath.hpp>

namespace mppi::critics
{

void SpatioTemporalCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(enabled_, "enabled", true);
  getParam(tracks_topic_, "tracks_topic", std::string("/dynamic_tracks_state"));

  // Parametri del termine 1 (collisione futura).
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 20.0f);
  getParam(collision_radius_, "collision_radius", 0.5f);
  getParam(vel_std_gain_, "vel_std_gain", 0.4f);
  getParam(min_ped_speed_, "min_ped_speed", 0.4f);
  getParam(max_vel_std_, "max_vel_std", 0.8f);
  getParam(max_track_age_s_, "max_track_age_s", 0.5f);
  getParam(prediction_horizon_s_, "prediction_horizon_s", 3.0f);
  getParam(max_radius_, "max_radius", 1.5f);
  getParam(worst_case_, "worst_case", true);

  // Forma del costo. Default "both": i due termini coprono i rispettivi punti ciechi.
  getParam(cost_mode_, "cost_mode", std::string("both"));
  getParam(ttc_weight_, "ttc_weight", 20.0f);
  getParam(ttc_power_, "ttc_power", 1.0f);
  getParam(ttc_epsilon_, "ttc_epsilon", 0.1f);

  use_overlap_ = (cost_mode_ == "overlap" || cost_mode_ == "both");
  use_ttc_ = (cost_mode_ == "inv_ttc" || cost_mode_ == "both");
  if (!use_overlap_ && !use_ttc_) {
    RCLCPP_WARN(
      logger_, "cost_mode '%s' non valido: ripiego su 'both'.", cost_mode_.c_str());
    cost_mode_ = "both";
    use_overlap_ = true;
    use_ttc_ = true;
  }

  // Diagnostica aggregata (0 = spenta).
  getParam(diag_period_calls_, "diag_period_calls", 0);

  // Parametri di visualizzazione.
  getParam(publish_predictions_, "publish_predictions", true);
  getParam(predictions_topic_, "predictions_topic",
    std::string("/spatiotemporal_critic/predictions"));
  // traiettoria robot: topic separato + flag proprio (default: sempre attiva)
  getParam(publish_robot_traj_, "publish_robot_trajectory", true);
  getParam(robot_traj_topic_, "robot_trajectory_topic",
    std::string("/spatiotemporal_critic/robot_trajectory"));
  // frame di pubblicazione (default = frame delle tracce = odom)
  getParam(world_frame_, "world_frame", std::string("odom"));

  auto node = parent_.lock();
  tracks_sub_ = node->create_subscription<dynamic_tracker_msgs::msg::TrackArray>(
    tracks_topic_, rclcpp::QoS(10),
    std::bind(&SpatioTemporalCritic::tracksCallback, this, std::placeholders::_1));

  if (publish_predictions_) {
    pred_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      predictions_topic_, rclcpp::QoS(1));
  }
  if (publish_robot_traj_) {
    robot_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      robot_traj_topic_, rclcpp::QoS(1));
  }

  RCLCPP_INFO(
    logger_,
    "SpatioTemporalCritic: mode=%s | overlap w=%.1f (worst_case=%d) | "
    "ttc w=%.1f pow=%.2f eps=%.2f | horizon=%.1fs radius=%.2f | tracks_topic=%s",
    cost_mode_.c_str(), weight_, static_cast<int>(worst_case_),
    ttc_weight_, ttc_power_, ttc_epsilon_, prediction_horizon_s_,
    collision_radius_, tracks_topic_.c_str());
}

void SpatioTemporalCritic::tracksCallback(
  const dynamic_tracker_msgs::msg::TrackArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(tracks_mutex_);
  tracks_.clear();
  tracks_.reserve(msg->tracks.size());
  for (const auto & t : msg->tracks) {
    tracks_.push_back(TrackSnapshot{t.x, t.y, t.vx, t.vy, t.pos_std, t.vel_std});
  }
  tracks_stamp_ = msg->header.stamp;
  tracks_frame_ = msg->header.frame_id;
}

void SpatioTemporalCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }

  // Forme delle traiettorie: servono anche alla sola viz del robot, quindi le
  // leggo e valido PRIMA di qualunque early-return legato ai pedoni.
  const auto & traj_x = data.trajectories.x;   // [batch x time]
  const auto & traj_y = data.trajectories.y;
  const float dt = data.model_dt;
  const size_t batch = traj_x.shape(0);
  const size_t time = traj_x.shape(1);

  if (batch == 0 || time == 0) {
    return;
  }
  if (traj_y.shape(0) != batch || traj_y.shape(1) != time) {
    return;
  }
  if (data.costs.shape(0) != batch) {
    return;
  }

  // === VIZ TRAIETTORIA ROBOT: SEMPRE, su topic dedicato ===
  // Pubblicata a ogni ciclo di controllo, indipendente dai pedoni. E' la media
  // del batch gia' calcolato: costo trascurabile (nessuna candidate ricostruita).
  if (publish_robot_traj_ && robot_pub_) {
    publishRobotTrajectory(data, dt, time);
  }

  // Copio le tracce E lo stamp sotto lock (arrivano su un altro thread).
  std::vector<TrackSnapshot> tracks;
  builtin_interfaces::msg::Time tracks_stamp;
  {
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    tracks = tracks_;
    tracks_stamp = tracks_stamp_;
  }
  // Il termine di collisione futura ha senso solo con pedoni: se non ce ne
  // sono, la viz del robot e' gia' stata pubblicata sopra, quindi esco.
  if (tracks.empty()) {
    return;
  }

  // --- scarto tracce stantie: age = now - stamp dell'ultimo TrackArray ---
  // Senza questo, se il tracker si ferma il critic propaga all'infinito la CV
  // sull'ultima traccia. La viz del robot e' gia' uscita sopra, quindi esco pulito.
  if (auto node = parent_.lock()) {
    const double age = (node->now() - rclcpp::Time(tracks_stamp)).seconds();
    if (age > static_cast<double>(max_track_age_s_)) {
      return;
    }
  }

  // Penalita' per traiettoria. std::vector (non xtensor): la scrittura finale
  // su data.costs va fatta elemento per elemento (operazioni xtensor su
  // riferimenti = causa di crash su questo Humble).
  std::vector<float> penalty(batch, 0.0f);

  // ================== TERMINE 1: COLLISIONE FUTURA ==================
  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }

  const size_t kNoHit = std::numeric_limits<size_t>::max();
  std::vector<float> track_pen(batch, 0.0f);     // OVERLAP: profondita' nella bolla
  std::vector<size_t> first_hit(batch, kNoHit);  // TTC: passo del primo contatto

  // accumulatori di diagnostica: qualche somma per ciclo, costo trascurabile
  double diag_sum_ovl = 0.0, diag_sum_ttc = 0.0;
  size_t diag_n_hit = 0, diag_n_tracks = 0;
  size_t diag_tau_min = kNoHit;

  for (const auto & tr : tracks) {
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;
    }
    if (tr.vel_std > max_vel_std_) {
      continue;
    }
    ++diag_n_tracks;

    std::fill(track_pen.begin(), track_pen.end(), 0.0f);
    std::fill(first_hit.begin(), first_hit.end(), kNoHit);

    for (size_t j = 0; j < time_pred; ++j) {
      const float t = static_cast<float>(j) * dt;
      const double px = tr.x + tr.vx * t;
      const double py = tr.y + tr.vy * t;
      double radius = collision_radius_ + tr.pos_std +
        vel_std_gain_ * tr.vel_std * t;
      if (radius > max_radius_) {
        radius = max_radius_;
      }
      const double r2 = radius * radius;

      for (size_t i = 0; i < batch; ++i) {
        const double ddx = traj_x(i, j) - px;
        const double ddy = traj_y(i, j) - py;
        const double d2 = ddx * ddx + ddy * ddy;
        if (d2 >= r2) {
          continue;
        }

        // TEMPO: j cresce in modo monotono, quindi la PRIMA volta che
        // entriamo qui e' gia' l'istante di primo contatto. Nessuna
        // ricerca aggiuntiva, nessun costo extra.
        if (first_hit[i] == kNoHit) {
          first_hit[i] = j;
        }

        // PROFONDITA': la sqrt serve solo all'overlap. Se il termine e'
        // spento la saltiamo -- e' il loop piu' interno (batch x time).
        if (!use_overlap_) {
          continue;
        }
        const float closeness =
          static_cast<float>(1.0 - std::sqrt(d2) / radius);
        if (worst_case_) {
          if (closeness > track_pen[i]) {
            track_pen[i] = closeness;
          }
        } else {
          track_pen[i] += closeness;
        }
      }
    }

    // Combinazione dei due termini, ciascuno col PROPRIO peso.
    // NB: i pesi si applicano QUI, non piu' alla fine: cost_weight governa
    // l'overlap, ttc_weight il termine temporale, e restano indipendenti.
    for (size_t i = 0; i < batch; ++i) {
      if (use_overlap_) {
        const float c = weight_ * track_pen[i];
        penalty[i] += c;
        diag_sum_ovl += c;
      }
      if (first_hit[i] != kNoHit) {
        ++diag_n_hit;
        if (first_hit[i] < diag_tau_min) {
          diag_tau_min = first_hit[i];
        }
        if (use_ttc_) {
          const float tau = static_cast<float>(first_hit[i]) * dt;
          float c = 1.0f / (tau + ttc_epsilon_);
          if (std::fabs(ttc_power_ - 1.0f) > 1e-6f) {
            c = std::pow(c, ttc_power_);   // percorso lento solo se esponente != 1
          }
          c *= ttc_weight_;
          penalty[i] += c;
          diag_sum_ttc += c;
        }
      }
    }
  }

  for (size_t i = 0; i < batch; ++i) {
    float p = penalty[i];
    if (power_ > 1) {
      p = std::pow(p, static_cast<float>(power_));
    }
    data.costs(i) += p;
  }

  // --- diagnostica aggregata (una riga ogni diag_period_calls_ chiamate) ---
  // data.costs qui contiene GIA' il contributo degli altri critic (questo e'
  // l'ultimo della lista), quindi il confronto penalita'/costo dice davvero
  // quanto pesa questo critic sul totale.
  if (diag_period_calls_ > 0 && (++score_calls_ % diag_period_calls_) == 0) {
    float pen_max = 0.0f, cost_max = 0.0f;
    for (size_t i = 0; i < batch; ++i) {
      if (penalty[i] > pen_max) {
        pen_max = penalty[i];
      }
      if (data.costs(i) > cost_max) {
        cost_max = data.costs(i);
      }
    }
    const double tau_min = (diag_tau_min == kNoHit)
      ? -1.0 : static_cast<double>(diag_tau_min) * dt;
    RCLCPP_INFO(
      logger_,
      "[STC] mode=%s tracce=%zu contatti=%zu/%zu tau_min=%.2fs | "
      "overlap=%.1f ttc=%.1f | mia penalita' max=%.1f su costo max=%.1f",
      cost_mode_.c_str(), diag_n_tracks, diag_n_hit, batch * diag_n_tracks,
      tau_min, diag_sum_ovl, diag_sum_ttc, pen_max, cost_max);
  }

  // Viz scie dei PEDONI (solo con pedoni), sul loro topic separato.
  if (publish_predictions_ && pred_pub_) {
    publishPredictions(tracks, dt, time);
  }
}

void SpatioTemporalCritic::publishPredictions(
  const std::vector<TrackSnapshot> & tracks,
  float dt, size_t time)
{
  visualization_msgs::msg::MarkerArray arr;

  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  int id = 1;
  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }
  const size_t step = std::max<size_t>(1, static_cast<size_t>(0.5f / dt));

  // === SCIE DEI PEDONI (dove sara' ogni pedone, verde->rosso) ===
  for (const auto & tr : tracks) {
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;
    }
    if (tr.vel_std > max_vel_std_) {
      continue;
    }
    for (size_t j = 0; j < time_pred; j += step) {
      const float t = static_cast<float>(j) * dt;
      const double px = tr.x + tr.vx * t;
      const double py = tr.y + tr.vy * t;
      double radius = collision_radius_ + tr.pos_std +
        vel_std_gain_ * tr.vel_std * t;
      if (radius > max_radius_) {
        radius = max_radius_;
      }

      visualization_msgs::msg::Marker m;
      m.header.frame_id = tracks_frame_;
      m.header.stamp = tracks_stamp_;
      m.ns = "ped_prediction";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      // frame_locked: RViz ri-trasforma col TF piu' recente a ogni frame, cosi'
      // il marker non viene scartato se la map->odom allo stamp non e' bufferizzata.
      m.frame_locked = true;
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      m.pose.position.x = px;
      m.pose.position.y = py;
      m.pose.position.z = 0.1;
      m.pose.orientation.w = 1.0;
      m.scale.x = 2.0 * radius;
      m.scale.y = 2.0 * radius;
      m.scale.z = 0.05;
      const float frac = static_cast<float>(j) / static_cast<float>(time_pred);
      m.color.r = frac;
      m.color.g = 1.0f - frac;
      m.color.b = 0.0f;
      m.color.a = 0.35f;
      arr.markers.push_back(m);
    }
  }

  pred_pub_->publish(arr);
}

void SpatioTemporalCritic::publishRobotTrajectory(
  const mppi::CriticData & data, float dt, size_t time)
{
  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const size_t batch = traj_x.shape(0);
  if (batch == 0) {
    return;
  }

  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }

  // Stamp fresco: questa viz e' pubblicata sempre, anche senza tracce recenti.
  rclcpp::Time stamp;
  auto node = parent_.lock();
  if (node) {
    stamp = node->now();
  }

  visualization_msgs::msg::MarkerArray arr;
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  // UNA linea (LINE_STRIP) che collega i punti MEDI del batch a ogni istante:
  // baricentro di dove il robot pensa di andare. NB: media delle traiettorie
  // campionate (~ nominale), non l'ottima esatta. Campiono ogni istante per una
  // linea liscia (batch*time e' trascurabile per la CPU).
  visualization_msgs::msg::Marker line;
  line.header.frame_id = world_frame_;   // "odom" di default
  line.header.stamp = stamp;
  line.ns = "robot_mean";
  line.id = 0;
  line.type = visualization_msgs::msg::Marker::LINE_STRIP;
  line.action = visualization_msgs::msg::Marker::ADD;
  line.frame_locked = true;
  line.lifetime = rclcpp::Duration::from_seconds(0.3);
  line.pose.orientation.w = 1.0;   // LINE_STRIP: i punti sono gia' in world, posa identita'
  line.scale.x = 0.05;             // spessore linea [m]
  line.color.r = 0.1f;
  line.color.g = 0.3f;
  line.color.b = 1.0f;
  line.color.a = 0.9f;

  for (size_t j = 0; j < time_pred; ++j) {
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < batch; ++i) {
      mx += traj_x(i, j);
      my += traj_y(i, j);
    }
    mx /= static_cast<double>(batch);
    my /= static_cast<double>(batch);

    geometry_msgs::msg::Point p;
    p.x = mx;
    p.y = my;
    p.z = 0.15;
    line.points.push_back(p);
  }

  arr.markers.push_back(line);
  robot_pub_->publish(arr);
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::SpatioTemporalCritic,
  mppi::critics::CriticFunction)