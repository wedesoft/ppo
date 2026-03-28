(ns ppo.pendulum
    (:gen-class)
    (:require [clojure.math :refer (to-radians)]
              [quil.core :as q]
              [quil.middleware :as m])
    (:import [java.util.concurrent CountDownLatch]))

(defn setup []
  {:angle 0.0
   :angle-velocity 0.0
   :length 200
   :origin [(/ (q/width) 2) (/ (q/height) 2)]
   :gravity 0.4})

(defn update-state [state]
  (let [mouse-x (q/mouse-x)
        [origin-x _origin-y] (:origin state)
        diff (- mouse-x origin-x)
        angle-acceleration (to-radians (* 10 (/ diff origin-x)))
        dt (/ 1.0 60.0)]
    (q/frame-rate 60)
    (-> state
        (update :angle-velocity #(+ % (* dt angle-acceleration)))
        (update :angle #(+ % (* dt (:angle-velocity state)))))))

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
  (let [latch (CountDownLatch. 1)]
    (q/sketch
      :title "Inverted Pendulum with Mouse Control"
      :size [500 500]
      :setup setup
      :update update-state
      :draw draw-state
      :middleware [m/fun-mode]
      :on-close (fn [& _] (.countDown latch)))
    (.await latch))
  (System/exit 0))
