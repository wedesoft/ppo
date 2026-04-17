(ns ppo.pendulum
    (:gen-class)
    (:require [clojure.math :refer (PI to-radians cos sin)]
              [clojure.core.async :as async]
              [quil.core :as q]
              [quil.middleware :as m]
              [libpython-clj2.require :refer (require-python)]
              [libpython-clj2.python :refer (py.) :as py]
              [ppo.mlp :refer (Actor tensor tolist)]
              [ppo.environment :refer (Environment)])
    (:import [java.util.concurrent CountDownLatch]))

(require-python '[torch :as torch])


(def frame-rate 20)


(def config
  {:length  (/ 2.0 3.0)
   :max-speed 8.0
   :motor 6.0
   :gravitation 10.0
   :dt (/ 1.0 frame-rate)
   :save false
   :timeout 10.0
   :target-angle 0.25
   :target-time 1.0
   :angle-weight 1.0
   :velocity-weight 0.1
   :control-weight 0.001})


(defn setup
  "Initialise pendulum"
  [angle velocity]
  {:angle          angle
   :velocity       velocity
   :t              0.0})


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


(defn up-deviation
  "Angular deviation from up angle"
  [angle]
  (- (mod angle (* 2 PI)) PI))


(defn at-target?
  "Decide whether pendulum is at target state"
  ([state]
   (at-target? state config))
  ([{:keys [angle]} {:keys [target-angle]}]
   (<= (abs (up-deviation angle)) target-angle)))


(defn update-state
  "Perform simulation step of pendulum"
  ([state action]
   (update-state state action config))
  ([{:keys [angle velocity t] :as state} {:keys [control]} {:keys [dt motor gravitation length max-speed]}]
   (let [gravity        (pendulum-gravity gravitation length angle)
         motor          (motor-acceleration control motor)
         t              (+ t dt)
         acceleration   (+ motor gravity)
         velocity       (max (- max-speed) (min max-speed (+ velocity (* acceleration dt))))
         angle          (+ angle (* velocity dt))]
     {:angle          angle
      :velocity       velocity
      :t              t})))


(defn observation
  "Get observation from state"
  [{:keys [angle velocity]} {:keys [max-speed]}]
  [(cos angle) (sin angle) (/ velocity max-speed)])


(defn action
  "Convert array to action"
  [array]
  {:control (max -1.0 (min 1.0 (- (* 2.0 (first array)) 1.0)))})


(defn truncate?
  "Decide whether a run should be aborted"
  ([state]
   (truncate? state config))
  ([{:keys [t]} {:keys [timeout]}]
   (>= t timeout)))


(defn done?
  "Decide whether pendulum achieved target state"
  ([state & _args]
   false))


(defn sqr
  "Square of number"
  [x]
  (* x x))


(defn reward
  "Reward function"
  [{:keys [angle velocity] :as state} {:keys [angle-weight velocity-weight control-weight final-reward] :as config} {:keys [control]}]
  (- (+ (* angle-weight (sqr (up-deviation angle)))
        (* velocity-weight (sqr velocity))
        (* control-weight (sqr control)))))


(defrecord Pendulum [config state]
  Environment
  (environment-update [_this input]
    (->Pendulum config (update-state state (action input) config)))
  (environment-observation [_this]
    (observation state config))
  (environment-done? [_this]
    (done? state config))
  (environment-truncate? [_this]
    (truncate? state config))
  (environment-reward [_this input]
    (reward state config (action input))))


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


(defn -main [& _args]
  (let [actor     (Actor 3 100 1)
        done-chan (async/chan)]
    (when (.exists (java.io.File. "actor.pt"))
      (py. actor load_state_dict (torch/load "actor.pt")))
    (q/sketch
      :title "Inverted Pendulum with Mouse Control"
      :size [854 480]
      :setup #(setup (- (rand 2.0) 1.0) 0.0)
      :update (fn [state]
                  (let [observation (observation state config)
                        action      (if (q/mouse-pressed?)
                                      {:control (- (/ (q/mouse-x) (/ (q/width) 2.0)) 1.0)}
                                      (action (tolist (py. actor deterministic_act (tensor observation)))))
                        reward      (reward state config action)
                        state       (update-state state action)]
                    (when (done? state) (async/close! done-chan))
                    state))
      :draw draw-state
      :middleware [m/fun-mode]
      :on-close (fn [& _] (async/close! done-chan)))
    (async/<!! done-chan))
  (System/exit 0))
