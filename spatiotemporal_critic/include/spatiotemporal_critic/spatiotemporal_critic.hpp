#ifndef SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_
#define SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_

#include <mutex>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "nav2_mppi_controller/critic_function.hpp"
#include "dynamic_tracker_msgs/msg/track_array.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace mppi::critics
{

struct TrackSnapshot
{
  double x, y;        // posizione al tempo di pubblicazione [m]
  double vx, vy;      // velocita' stimata [m/s]
  double pos_std;     // incertezza posizione [m]
  double vel_std;     // incertezza velocita' [m/s]
};

class SpatioTemporalCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;

  // Chiamato a ogni ciclo di controllo (solo con un goal attivo).
  // NB: CriticData sta nel namespace mppi::, NON mppi::critics::
  void score(mppi::CriticData & data) override;

protected:
  // Callback del topic /dynamic_tracks_state: aggiorna lo snapshot delle tracce.
  void tracksCallback(const dynamic_tracker_msgs::msg::TrackArray::SharedPtr msg);

  // Pubblica le scie predette dei PEDONI (per RViz), sul topic dei pedoni.
  void publishPredictions(
    const std::vector<TrackSnapshot> & tracks,
    float dt, size_t time);

  // Pubblica la traiettoria del ROBOT (media del batch), su un topic SEPARATO
  // e a OGNI ciclo di controllo, indipendente dalla presenza di pedoni.
  // NB: e' la MEDIA delle traiettorie campionate (~ nominale), non l'ottima
  // esatta di MPPI; e' un ripiego leggero (nessun calcolo extra: media il batch
  // gia' prodotto dall'ottimizzatore) al posto di visualize=true.
  void publishRobotTrajectory(
    const mppi::CriticData & data, float dt, size_t time);

  // --- parametri (letti in initialize) ---
  bool enabled_{true};
  std::string tracks_topic_;

  // --- termine 1: collisione futura ---
  unsigned int power_{1};          // esponente della penalita' (come gli altri critic)
  float weight_{20.0f};            // peso del termine di collisione futura
  float collision_radius_{0.5f};   // [m] raggio base di sicurezza robot+pedone
  float vel_std_gain_{0.4f};       // quanto l'incertezza di velocita' allarga l'alone nel tempo
  float min_ped_speed_{0.4f};      // [m/s] sotto: pedone "fermo", non genera collisione futura
  float max_vel_std_{0.8f};        // [m/s] scarta tracce con velocita' troppo incerta
  float max_track_age_s_{0.5f};    // [s] tracce piu' vecchie di cosi' vengono ignorate

  // --- limiti della predizione (per non generare aloni giganti) ---
  float prediction_horizon_s_{3.0f};  // [s] predici il pedone solo per questo tempo
  float max_radius_{1.5f};            // [m] tetto massimo dell'alone di sicurezza

  // Come si aggrega la vicinanza al pedone lungo il tempo.
  //   true  (WORST CASE): solo l'avvicinamento PEGGIORE della traiettoria.
  //   false (SOMMA): accumula a ogni istante (utile in tesi come A/B).
  bool worst_case_{true};

  // --- visualizzazione (RViz) ---
  bool publish_predictions_{true};        // pubblica le scie predette dei pedoni
  std::string predictions_topic_;         // topic del MarkerArray dei pedoni
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pred_pub_;

  // traiettoria del robot su topic SEPARATO, pubblicata sempre (debug in RViz)
  bool publish_robot_traj_{true};
  std::string robot_traj_topic_;          // topic del MarkerArray del robot
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr robot_pub_;

  std::string world_frame_;               // frame in cui pubblicare (= frame tracce)

  // --- stato condiviso tra la callback e score() ---
  std::mutex tracks_mutex_;
  std::vector<TrackSnapshot> tracks_;
  builtin_interfaces::msg::Time tracks_stamp_;
  std::string tracks_frame_{"odom"};   // frame dell'ultimo TrackArray ricevuto

  rclcpp::Subscription<dynamic_tracker_msgs::msg::TrackArray>::SharedPtr tracks_sub_;

  unsigned int score_calls_{0};    // contatore per il log diagnostico
};

}  // namespace mppi::critics

#endif  // SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_