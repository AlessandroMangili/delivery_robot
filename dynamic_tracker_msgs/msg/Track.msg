int32   id            # identificativo stabile della traccia (per il debug/associazione)
float64 x             # [m] posizione stimata
float64 y             # [m]
float64 vx            # [m/s] velocita' stimata (dal Kalman)
float64 vy            # [m/s]
float64 pos_std       # [m]   incertezza sulla POSIZIONE (sqrt della cov posizione)
float64 vel_std       # [m/s] incertezza sulla VELOCITA' (sqrt della cov velocita')
                      #       il critic allarga l'alone di sicurezza in proporzione a
                      #       questa: un pedone appena visto e' meno prevedibile.
