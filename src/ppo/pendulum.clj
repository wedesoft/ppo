(ns ppo.pendulum
    (:gen-class)
    (:require [clojure.math :refer (to-radians cos sin)]
              [clojure.core.async :as async]
              [quil.core :as q]
              [quil.middleware :as m])
    (:import [java.util.concurrent CountDownLatch]))


(def frame-rate 25)


(def config
  {:length  0.8
   :friction 0.1
   :motor 5.0
   :gravitation 20.0
   :dt (/ 1.0 frame-rate)
   :save false})


(defn setup
  "Initialise pendulum"
  [angle]
  {:angle    angle
   :velocity 0.0
   :t        0.0})


(defn pendulum-gravity
  "Determine angular acceleration due to gravity"
  [gravitation length angle]
  (/ (* (sin angle) (- gravitation)) length))


(defn motor-acceleration
  "Angular acceleration from motor"
  [control motor-acceleration]
  (* control motor-acceleration))


(defn sign
  "Get sign of number"
  [x]
  (cond
    (pos? x) 1
    (neg? x) -1
    :else 0))


(defn friction-acceleration
  "Angular acceleration due to friction"
  [friction velocity dt]
  (* (min friction (/ (abs velocity) dt)) (- (sign velocity))))


(defn update-state
  "Perform simulation step of pendulum"
  ([state action]
   (update-state state action config))
  ([{:keys [angle velocity t]} {:keys [control]} {:keys [dt friction motor gravitation length]}]
   (let [friction     (friction-acceleration friction velocity dt)
         gravity      (pendulum-gravity gravitation length angle)
         motor        (motor-acceleration control motor)
         t            (+ t dt)
         acceleration (+ motor gravity friction)
         velocity     (+ velocity (* acceleration dt))
         angle        (+ angle (* velocity dt))]
     {:angle    angle
      :velocity velocity
      :t        t})))


(defn observation
  "Get observation from state"
  [{:keys [angle velocity]}]
  (double-array [angle velocity]))


(defn action
  "Convert array to action"
  [array]
  {:control (aget array 0)})


(defn draw-state [{:keys [angle]}]
  (let [origin-x   (/ (q/width) 2)
        origin-y   (/ (q/height) 2)
        length     (* 0.5 (q/height) (:length config))
        pendulum-x (+ origin-x (* length (sin angle)))
        pendulum-y (+ origin-y (* length (cos angle)))
        size       (* 0.05 (q/height))]
    (q/frame-rate frame-rate)
    (q/background 255)
    ; set thickness of line
    (q/stroke-weight 5)
    (q/stroke 0)
    (q/fill 175)
    (q/line origin-x origin-y pendulum-x pendulum-y)
    (q/stroke-weight 1)
    (q/ellipse pendulum-x pendulum-y size size)
    (when (:save config)
      (q/save-frame "frame-####.png"))))


(defn mouse-action
  "Control motor with mouse"
  []
  {:control (if (q/mouse-pressed?) (- (/ (q/mouse-x) (/ (q/width) 2.0)) 1.0) 0.0)})


(defn -main [& _args]
  (let [done-chan (async/chan)]
    (q/sketch
      :title "Inverted Pendulum with Mouse Control"
      :size [854 480]
      :setup #(setup 0.1)
      :update #(update-state % (mouse-action))
      :draw draw-state
      :middleware [m/fun-mode]
      :on-close (fn [& _] (async/close! done-chan)))
    (async/<!! done-chan))
  (System/exit 0))
