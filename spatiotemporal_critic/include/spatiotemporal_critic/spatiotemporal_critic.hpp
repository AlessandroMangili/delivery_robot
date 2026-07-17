// Critic spazio-temporale per MPPI (Nav2 Humble).
//
// PASSO 3 (questo file): solo lo SCHELETRO. Il critic si carica come plugin,
// legge i parametri e si iscrive al topic delle tracce, ma in score() NON
// aggiunge ancora alcun costo. Serve a verificare che compili e che MPPI lo
// carichi senza errori PRIMA di metterci la logica (passi 4 e 5).
//
// Eredita da mppi::critics::CriticFunction, l'interfaccia dei critic MPPI su
// Humble: initialize() una volta all'avvio, score() a ogni ciclo di controllo
// su tutte le traiettorie campionate.

#ifndef SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_
#define SPATIOTEMPORAL_CRITIC__SPATIOTEMPORAL_CRITIC_HPP_

#include <mutex>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

// critic_function.hpp e' la classe base: porta con se' il tipo CriticData
// (e' il parametro di score()). Non serve includerlo a parte -- e su Humble
// il path 'models/critics_data.hpp' non esiste comunque.
#include "nav2_mppi_controller/critic_function.hpp"
#include "dynamic_tracker_msgs/msg/track_array.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// La classe DEVE stare nel namespace mppi::critics: MPPI antepone questo
// namespace al nome scritto nello YAML (es. "SpatioTemporalCritic" ->
// cerca mppi::critics::SpatioTemporalCritic). Se sta in un namespace mio,
// pluginlib la registra ma MPPI non la trova.
namespace mppi::critics
{

// Copia locale e leggera di una traccia, protetta da mutex perche' arriva su un
// thread diverso (la callback del subscriber) da quello che gira score().
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
  // Chiamato una volta all'avvio: legge i parametri e crea il subscriber.
  void initialize() override;

  // Chiamato a ogni ciclo di controllo: qui, nei passi 4-5, aggiungeremo il
  // costo di collisione futura e di spazio libero. Per ora non fa nulla.
  // NB: CriticData sta nel namespace mppi::, NON mppi::critics::
  // (la classe base CriticFunction e' in mppi::critics ma il tipo del
  // parametro e' mppi::CriticData).
  void score(mppi::CriticData & data) override;

protected:
  // Callback del topic /dynamic_tracks_state: aggiorna lo snapshot delle tracce.
  void tracksCallback(const dynamic_tracker_msgs::msg::TrackArray::SharedPtr msg);

  // Costruisce e pubblica le scie predette dei pedoni E la traiettoria media
  // del robot (per RViz), cosi' si vede il confronto nel tempo.
  void publishPredictions(
    const std::vector<TrackSnapshot> & tracks,
    const mppi::CriticData & data,
    float dt, size_t time);

  // --- parametri (letti in initialize) ---
  bool enabled_{true};
  std::string tracks_topic_;

  // --- termine 1: collisione futura ---
  unsigned int power_{1};          // esponente della penalita' (come gli altri critic)
  float weight_{20.0f};            // peso del termine di collisione futura
  float collision_radius_{0.5f};   // [m] raggio base di sicurezza robot+pedone
  float vel_std_gain_{0.4f};       // quanto l'incertezza di velocita' allarga l'alone nel tempo
  float min_ped_speed_{0.4f};      // [m/s] sotto: pedone "fermo", non genera collisione futura
                                   //       (lo gestisce il cono statico gia' presente)
  float max_vel_std_{0.8f};        // [m/s] scarta tracce con velocita' troppo incerta:
                                   //       appena nate o rumorose (spesso statici mal
                                   //       classificati). Filtro anti-falsi-positivi.
  float max_track_age_s_{0.5f};    // [s] tracce piu' vecchie di cosi' vengono ignorate

  // --- limiti della predizione (per non generare aloni giganti) ---
  float prediction_horizon_s_{3.0f};  // [s] predici il pedone solo per questo tempo,
                                      //     anche se l'orizzonte MPPI e' piu' lungo.
                                      //     Un pedone non e' prevedibile oltre ~3s.
  float max_radius_{1.5f};            // [m] tetto massimo dell'alone di sicurezza

  // --- visualizzazione (RViz) ---
  bool publish_predictions_{true};        // pubblica le scie predette dei pedoni
  std::string predictions_topic_;         // topic del MarkerArray
  std::string world_frame_;               // frame in cui pubblicare (= frame tracce)
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pred_pub_;

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