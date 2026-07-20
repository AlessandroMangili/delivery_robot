// Critic spazio-temporale per MPPI (Nav2 Humble).
//
// TERMINE 1 - collisione futura: confronta dove sara' il pedone e dove sara' il
//   robot allo STESSO istante t, e penalizza l'incontro. E' cio' che il cono
//   statico non puo' fare, perche' non sa QUANDO il robot passera' di li'.
//   La scelta di DOVE scartare (destra o sinistra) e' lasciata a MPPI e ai
//   critic di costo gia' presenti: qui diciamo solo dove NON si puo' andare.

#include "spatiotemporal_critic/spatiotemporal_critic.hpp"

#include <cmath>
#include <algorithm>
#include <utility>

#include <xtensor/xmath.hpp>

// namespace mppi::critics: vedi la nota nell'header. MPPI cerca il critic
// come mppi::critics::<nome-nello-yaml>.
namespace mppi::critics
{

void SpatioTemporalCritic::initialize()
{
  // getParam appartiene a CriticFunction: legge i parametri sotto il nome di
  // questo critic nel controller_server (es. FollowPath.SpatioTemporalCritic.*).
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

  // Parametri di visualizzazione.
  getParam(publish_predictions_, "publish_predictions", true);
  getParam(predictions_topic_, "predictions_topic",
    std::string("/spatiotemporal_critic/predictions"));

  // Il subscriber vive sul nodo del controller_server (parent_ e' un weak_ptr
  // al nav2 lifecycle node). QoS 10, best-effort va bene per tracce a 10 Hz.
  auto node = parent_.lock();
  tracks_sub_ = node->create_subscription<dynamic_tracker_msgs::msg::TrackArray>(
    tracks_topic_, rclcpp::QoS(10),
    std::bind(&SpatioTemporalCritic::tracksCallback, this, std::placeholders::_1));

  if (publish_predictions_) {
    pred_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      predictions_topic_, rclcpp::QoS(1));
  }

  RCLCPP_INFO(
    logger_,
    "SpatioTemporalCritic avviato (termine 1: collisione futura). "
    "enabled=%d, weight=%.1f, collision_radius=%.2f, min_ped_speed=%.2f, tracks_topic=%s",
    static_cast<int>(enabled_), weight_, collision_radius_, min_ped_speed_,
    tracks_topic_.c_str());
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

  // Copio le tracce sotto lock (arrivano su un altro thread).
  std::vector<TrackSnapshot> tracks;
  {
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    tracks = tracks_;
  }
  if (tracks.empty()) {
    return;
  }

  const auto & traj_x = data.trajectories.x;   // [batch x time]
  const auto & traj_y = data.trajectories.y;
  const float dt = data.model_dt;
  const size_t batch = traj_x.shape(0);
  const size_t time = traj_x.shape(1);

  // Guardie sulle forme: se qualcosa non torna, non tocchiamo la memoria.
  if (batch == 0 || time == 0) {
    return;
  }
  if (traj_y.shape(0) != batch || traj_y.shape(1) != time) {
    return;
  }
  if (data.costs.shape(0) != batch) {
    return;
  }

  // Penalita' per traiettoria. std::vector (non xtensor): la scrittura finale
  // su data.costs va fatta elemento per elemento, non con operazioni
  // vettoriali xtensor su riferimenti (causa di crash su questo Humble).
  std::vector<float> penalty(batch, 0.0f);

  // ================== TERMINE 1: COLLISIONE FUTURA ==================
  // Salta se non ci sono pedoni, ma NON esce dalla funzione: il termine 2
  // (spazio libero) deve funzionare comunque, anche a marciapiede vuoto.
  // Limito la predizione a prediction_horizon_s: oltre, un pedone non e'
  // prevedibile e l'alone diventerebbe enorme, coprendo tutto e togliendo
  // al robot ogni via d'uscita.
  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }

  // Penalita' della SINGOLA traccia, riusata a ogni giro. In modalita'
  // worst_case tiene il massimo (avvicinamento peggiore); altrimenti somma.
  std::vector<float> track_pen(batch, 0.0f);

  for (const auto & tr : tracks) {
    // solo pedoni in MOVIMENTO: i fermi sono gia' ostacoli fisici nella
    // costmap (obstacle_layer + inflation), e la predizione lineare su un
    // fermo non ha senso
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;
    }
    // scarta tracce con velocita' troppo incerta (appena nate / rumorose /
    // statici mal classificati): la loro predizione sarebbe inaffidabile
    if (tr.vel_std > max_vel_std_) {
      continue;
    }

    std::fill(track_pen.begin(), track_pen.end(), 0.0f);

    // per ogni istante j (fino all'orizzonte di predizione): dove sara' il
    // pedone e dove sara' il robot ALLO STESSO t. Se vicini -> penalita'.
    for (size_t j = 0; j < time_pred; ++j) {
      const float t = static_cast<float>(j) * dt;
      const double px = tr.x + tr.vx * t;      // pedone al tempo t (Kalman lineare)
      const double py = tr.y + tr.vy * t;
      // alone di sicurezza: base + incertezza di partenza + incertezza di
      // velocita' che cresce nel tempo, MA con un tetto massimo per non
      // coprire tutto il marciapiede
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
        if (d2 < r2) {
          // 1 al centro dell'alone, 0 al bordo: sfiorare costa poco,
          // centrare in pieno costa molto
          const float closeness =
            static_cast<float>(1.0 - std::sqrt(d2) / radius);
          if (worst_case_) {
            // tengo solo l'avvicinamento PEGGIORE: "quanto vicino ci passi".
            // Cosi' superare (stargli accanto a distanza per qualche secondo)
            // costa poco, mentre andargli addosso costa molto.
            if (closeness > track_pen[i]) {
              track_pen[i] = closeness;
            }
          } else {
            // somma: misura "quanto tempo" resti vicino. Punisce il sorpasso
            // quanto l'impatto -> il robot preferisce tornare indietro.
            track_pen[i] += closeness;
          }
        }
      }
    }

    // le tracce si sommano tra loro: due pedoni minacciosi pesano piu' di uno
    for (size_t i = 0; i < batch; ++i) {
      penalty[i] += track_pen[i];
    }
  }

  // Scrittura del termine 1, elemento per elemento.
  // NB: l'esponente si applica alla penalita' NORMALIZZATA (che in modalita'
  // worst_case sta in [0,1]), poi si moltiplica per il peso. Cosi' cost_power
  // modella la FORMA della spaziatura personale senza stravolgere la scala:
  //   power 1 -> penalita' lineare con la vicinanza;
  //   power 2 -> sfiorare il bordo dell'alone costa poco, avvicinarsi davvero
  //              costa molto di piu' (profilo piu' simile allo spazio personale
  //              descritto in letteratura). Con power>1 alza anche cost_weight.
  for (size_t i = 0; i < batch; ++i) {
    float p = penalty[i];
    if (power_ > 1) {
      p = std::pow(p, static_cast<float>(power_));
    }
    data.costs(i) += p * weight_;
  }

  // Visualizzazione per RViz: le scie predette dei pedoni + traiettoria robot.
  if (publish_predictions_ && pred_pub_) {
    publishPredictions(tracks, data, dt, time);
  }
}

void SpatioTemporalCritic::publishPredictions(
  const std::vector<TrackSnapshot> & tracks,
  const mppi::CriticData & data,
  float dt, size_t time)
{
  visualization_msgs::msg::MarkerArray arr;

  // Marker 0: cancella i marker del ciclo precedente (evita scie fantasma
  // quando un pedone scompare).
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  int id = 1;
  // Stesso orizzonte di predizione usato in score().
  size_t time_pred = time;
  if (dt > 0.0f) {
    const size_t h = static_cast<size_t>(prediction_horizon_s_ / dt);
    time_pred = std::min(time, std::max<size_t>(1, h));
  }
  // Campiono alcuni istanti (non tutti: sarebbero troppe sfere). Uno ogni ~0.5 s.
  const size_t step = std::max<size_t>(1, static_cast<size_t>(0.5f / dt));

  // === 1) SCIE DEI PEDONI (dove sara' ogni pedone, verde->rosso) ===
  for (const auto & tr : tracks) {
    const double speed = std::hypot(tr.vx, tr.vy);
    if (speed < min_ped_speed_) {
      continue;   // i fermi non vengono predetti dal critic
    }
    if (tr.vel_std > max_vel_std_) {
      continue;   // stesso filtro di score(): tracce troppo incerte scartate
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
      // lifetime: se il critic smette di pubblicare (robot fermo, pedone
      // sparito), RViz cancella da solo il marker dopo 0.3s invece di
      // lasciarlo bloccato a schermo.
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      m.pose.position.x = px;
      m.pose.position.y = py;
      m.pose.position.z = 0.1;
      m.pose.orientation.w = 1.0;
      // diametro = alone di sicurezza (cresce nel tempo, con tetto)
      m.scale.x = 2.0 * radius;
      m.scale.y = 2.0 * radius;
      m.scale.z = 0.05;
      // colore: verde vicino nel tempo -> rosso lontano nel futuro
      const float frac = static_cast<float>(j) / static_cast<float>(time_pred);
      m.color.r = frac;
      m.color.g = 1.0f - frac;
      m.color.b = 0.0f;
      m.color.a = 0.35f;
      arr.markers.push_back(m);
    }
  }

  // === 2) TRAIETTORIA MEDIA DEL ROBOT (dove sara' il robot, in BLU) ===
  // MPPI valuta 2000 traiettorie; disegnarle tutte sarebbe illeggibile.
  // Mostro la MEDIA del batch a ogni istante: e' il "baricentro" di dove il
  // robot sta pensando di andare. Cosi' vedi il confronto: al tempo t, il
  // punto blu (robot) e' dentro il cerchio del pedone allo stesso t? -> collisione.
  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const size_t batch = traj_x.shape(0);
  if (batch > 0) {
    for (size_t j = 0; j < time_pred; j += step) {
      // media su tutte le traiettorie a questo istante
      double mx = 0.0, my = 0.0;
      for (size_t i = 0; i < batch; ++i) {
        mx += traj_x(i, j);
        my += traj_y(i, j);
      }
      mx /= static_cast<double>(batch);
      my /= static_cast<double>(batch);

      visualization_msgs::msg::Marker m;
      m.header.frame_id = tracks_frame_;
      m.header.stamp = tracks_stamp_;
      m.ns = "robot_prediction";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      m.pose.position.x = mx;
      m.pose.position.y = my;
      m.pose.position.z = 0.15;
      m.pose.orientation.w = 1.0;
      // sfere piccole e blu: e' un percorso, non un alone
      m.scale.x = 0.18;
      m.scale.y = 0.18;
      m.scale.z = 0.05;
      m.color.r = 0.1f;
      m.color.g = 0.3f;
      m.color.b = 1.0f;
      m.color.a = 0.9f;
      arr.markers.push_back(m);
    }
  }

  pred_pub_->publish(arr);
}

}  // namespace mppi::critics

// Registra la classe come plugin pluginlib, cosi' MPPI la puo' caricare dal
// nome indicato nel YAML del controller.
#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mppi::critics::SpatioTemporalCritic,
  mppi::critics::CriticFunction)