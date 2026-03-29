(ns ppo.pendulum
    (:gen-class)
    (:require [clojure.math :refer (to-radians sin)]
              [clojure.core.async :as async]
              [quil.core :as q]
              [quil.middleware :as m])
    (:import [java.util.concurrent CountDownLatch]))

(defn setup []
  {:angle 1.0
   :angle-velocity 0.0
   :length 200
   :origin [(/ (q/width) 2) (/ (q/height) 2)]
   :motor 20.0
   :friction 0.02
   :gravity 5.0})

(defn update-state [state]
  (let [mouse-x (q/mouse-x)
        [origin-x _origin-y] (:origin state)
        diff (- mouse-x origin-x)
        motor-acceleration (to-radians (* (:motor state) (/ diff origin-x)))
        friction-acceleration (- (* (:angle-velocity state) (:friction state)))
        gravity-acceleration (- (* (:gravity state) (sin (:angle state))))
        acceleration (+ gravity-acceleration motor-acceleration friction-acceleration)
        dt (/ 1.0 60.0)
        angle-velocity (+ (:angle-velocity state) (* dt acceleration))
        angle (+ (:angle state) (* dt angle-velocity))]
    (q/frame-rate 60)
    (assoc state
           :angle-velocity angle-velocity
           :angle angle)))

(defn draw-state [{:keys [angle length origin]}]
  (q/background 255)
  (let [[origin-x origin-y] origin
        x (+ origin-x (* length (Math/sin angle)))
        y (+ origin-y (* length (Math/cos angle)))]
    (q/stroke 0)
    (q/fill 175)
    (q/line origin-x origin-y x y)
    (q/ellipse x y 30 30)))

(defn -main [& _args]
  (let [done-chan (async/chan)]
    (q/sketch
      :title "Inverted Pendulum with Mouse Control"
      :size [500 500]
      :setup setup
      :update update-state
      :draw draw-state
      :middleware [m/fun-mode]
      :on-close (fn [& _] (async/close! done-chan)))
    (async/<!! done-chan))
  (System/exit 0))
