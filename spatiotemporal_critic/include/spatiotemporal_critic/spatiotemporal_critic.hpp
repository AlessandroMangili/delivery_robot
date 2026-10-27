#ifndef SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_
#define SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_

#include <fstream>
#include <mutex>
#include <string>
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

  // Posa e velocita' attuali del robot, per il filtro dei pedoni dietro.
  struct RobotRef
  {
    double x, y, cyaw, syaw, v;   // v: velocita' lungo la direzione di marcia [m/s]
  };

  // true se il pedone in (px,py) all'istante t va considerato: davanti al
  // robot fatto avanzare a velocita' costante fino a t (oltre -rear_margin_
  // lungo la direzione di marcia) oppure entro rear_keep_radius_ da lui.
  // Sempre true con rear_filter_ spento.
  bool considerato(double px, double py, double t, const RobotRef & r) const;

  // Pubblica le scie predette dei PEDONI (per RViz), sul topic dei pedoni.
  // Solo gli istanti considerati dal critic.
  void publishPredictions(
    const std::vector<TrackSnapshot> & tracks,
    float dt, size_t time, const RobotRef & r);

  // Pubblica il campione a costo minimo del batch (debug del critic), su un
  // topic SEPARATO, a ogni chiamata di score(). Va chiamata a costi definitivi.
  // La traiettoria eseguita da MPPI e' invece "Optimal Trajectory" su
  // /trajectories (visualize: true).
  void publishRobotTrajectory(
    const mppi::CriticData & data, float dt, size_t time);

  // --- parametri (letti in initialize) ---
  bool enabled_{true};
  std::string tracks_topic_;

  // --- termine 1: collisione futura ---
  unsigned int power_{1};          // esponente della penalita' (come gli altri critic)
  float weight_{20.0f};            // peso del termine OVERLAP (closeness)
  float collision_radius_{0.5f};   // [m] raggio base di sicurezza robot+pedone
  float vel_std_gain_{0.4f};       // quanto l'incertezza di velocita' allarga l'alone nel tempo
  float min_ped_speed_{0.4f};      // [m/s] sotto: pedone "fermo", non genera collisione futura
  float max_vel_std_{0.8f};        // [m/s] scarta tracce con velocita' troppo incerta
  float max_track_age_s_{0.5f};    // [s] tracce piu' vecchie di cosi' vengono ignorate

  // --- pedoni dietro al robot ---
  // Si considerano solo gli istanti della predizione in cui il pedone e'
  // davanti al robot: chi segue o sorpassa e' responsabile di evitarlo, e
  // prima faceva scansare il robot. Resta tracciato, ed entra nel costo
  // appena la predizione lo porta davanti (sorpasso, attraversamento).
  bool rear_filter_{false};        // false = tutti i pedoni, come prima
  // false = decide la posizione ATTUALE: un pedone dietro e' solo tracciato,
  //         e appena supera il fianco del robot entra con tutta la predizione.
  // true  = decide la predizione: conta gia' ora la parte di scia che lo porta
  //         davanti (anticipa i sorpassi, ma disegna scie di pedoni ancora dietro).
  bool rear_use_prediction_{false};
  float rear_margin_m_{0.3f};      // [m] "davanti" parte da qui dietro il centro: include chi e' di lato
  bool rear_keep_close_{true};     // true = entro rear_keep_radius_m_ il pedone conta anche se dietro
  float rear_keep_radius_m_{1.0f}; // [m] raggio dell'eccezione sopra
  // Retromarcia: le traiettorie campionate che portano il robot dietro la
  // posizione attuale considerano TUTTI i pedoni (si verifica che dietro non
  // ci sia nessuno PRIMA di scegliere di arretrare). Niente regola "robot gia'
  // in retromarcia": con vx_min < 0 MPPI arretra spesso di pochi cm/s per
  // frenare, e spegneva il filtro. Le traiettorie in avanti non raggiungono
  // chi sta dietro.
  bool rear_on_reverse_{true};

  // --- limiti della predizione (per non generare aloni giganti) ---
  float prediction_horizon_s_{3.0f};  // [s] predici il pedone solo per questo tempo
  float max_radius_{1.5f};            // [m] tetto massimo dell'alone di sicurezza

  // Come si aggrega la vicinanza al pedone lungo il tempo.
  //   true  (WORST CASE): solo l'avvicinamento PEGGIORE della traiettoria.
  //   false (SOMMA): accumula a ogni istante (utile in tesi come A/B).
  bool worst_case_{true};

  // --- selezione della forma del costo (A/B da YAML, senza ricompilare) ---
  //   "overlap" = solo closeness: QUANTO a fondo entri nella bolla del pedone
  //   "inv_ttc" = solo 1/tau: QUANDO la bolla ti tocca la prima volta
  //   "both"    = entrambi, con pesi separati (default: coprono punti ciechi diversi)
  std::string cost_mode_;
  bool use_overlap_{true};
  bool use_ttc_{true};

  // --- termine inverse-TTC ---
  float ttc_weight_{20.0f};    // peso del termine 1/tau (indipendente da weight_)
  float ttc_power_{1.0f};      // esponente: 1 = graduale, 2 = concentrato sull'imminente
  float ttc_epsilon_{0.1f};    // [s] evita la divisione per zero quando tau = 0.
                               // Fissa anche il tetto: costo max = (1/eps)^ttc_power

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

  // --- diagnostica aggregata ---
  // 0 = spenta. N = una riga di log ogni N chiamate a score()
  // (a controller_frequency 10 Hz, N=10 -> circa un log al secondo).
  unsigned int diag_period_calls_{0};
  unsigned int score_calls_{0};    // contatore per il log diagnostico
  // Traccia CSV per i grafici: vuoto = disattivata. Una riga per ciclo.
  std::string diag_csv_path_;
  std::ofstream diag_csv_;
  // Righe [STC] su file (<diag_log_dir>/gg-mm-aa_HH-MM-SS.log) invece che nel
  // terminale. Vuoto = nel terminale, come prima.
  std::string diag_log_dir_;
  std::ofstream diag_log_;
};

}  // namespace mppi::critics

#endif  // SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_