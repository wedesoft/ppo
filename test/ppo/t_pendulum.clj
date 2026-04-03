(ns ppo.t-pendulum
    (:require
      [clojure.math :refer (PI cos)]
      [midje.sweet :refer :all]
      [ppo.pendulum :refer :all]
      [ppo.environment :refer :all]))


(fact "Set up pendulum"
      (setup (/ PI 2)) => {:angle (/ PI 2) :velocity 0.0 :t 0.0})


(facts "Angular acceleration due to gravitation"
       (pendulum-gravity 0.0 1.0 0.0) => 0.0
       (pendulum-gravity 9.81 1.0 (/ PI 2)) => -9.81
       (pendulum-gravity 9.81 2.0 (/ PI 2)) => -4.905
       (pendulum-gravity 9.81 1.0 0.0) => 0.0)


(facts "Angular acceleration from motor"
       (motor-acceleration 0.0 0.0) => 0.0
       (motor-acceleration 1.0 3.0) => 3.0
       (motor-acceleration 0.0 3.0) => 0.0)


(facts "Test sign of number"
       (sign 0.0) => 0
       (sign 1.0) => 1
       (sign 2.0) => 1
       (sign -1.0) => -1
       (sign -2.0) => -1)


(facts "Angular acceleration due to friction"
       (friction-acceleration 0.0 0.0 1.0) => 0.0
       (friction-acceleration 0.5 1.0 1.0) => -0.5
       (friction-acceleration 0.5 2.0 1.0) => -0.5
       (friction-acceleration 0.5 -1.0 1.0) => 0.5
       (friction-acceleration 0.5 1.0 4.0) => -0.25)


(facts "Update state"
       (update-state {:angle 0.0 :velocity 0.0 :t 0.0}
                     {:control 0.0} {:dt 1.0 :friction 0.0 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle 0.0 :velocity 0.0 :t 1.0}
       (update-state {:angle 0.0 :velocity 0.0 :t 2.0}
                     {:control 0.0} {:dt 1.0 :friction 0.0 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle 0.0 :velocity 0.0 :t 3.0}
       (update-state {:angle 0.0 :velocity 0.1 :t 0.0}
                     {:control 0.0} {:dt 1.0 :friction 0.0 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle 0.1 :velocity 0.1 :t 1.0}
       (update-state {:angle 0.0 :velocity 0.1 :t 0.0}
                     {:control 0.0} {:dt 0.5 :friction 0.0 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle 0.05 :velocity 0.1 :t 0.5}
       (update-state {:angle 0.0 :velocity 1.0 :t 0.0}
                     {:control 0.0} {:dt 1.0 :friction 0.5 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle 0.5 :velocity 0.5 :t 1.0}
       (update-state {:angle 0.0 :velocity 1.0 :t 0.0}
                     {:control 0.0} {:dt 1.0 :friction 2.0 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle 0.0 :velocity 0.0 :t 1.0}
       (update-state {:angle (/ PI 2) :velocity 0.0 :t 0.0}
                     {:control 0.0} {:dt 1.0 :friction 0.0 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle (- (/ PI 2) 9.81) :velocity -9.81 :t 1.0}
       (update-state {:angle 0.0 :velocity 0.0 :t 0.0}
                     {:control 1.0} {:dt 1.0 :friction 0.0 :gravitation 9.81 :length 1.0 :motor 2.0})
       => {:angle 2.0 :velocity 2.0 :t 1.0})


(facts "Get observation array from state"
       (observation {:angle 0.0 :velocity 0.0}) => [1.0 0.0 0.0]
       (observation {:angle 0.0 :velocity 0.5}) => [1.0 0.0 0.5]
       (observation {:angle (/ PI 2) :velocity 0.0}) => [(cos (/ PI 2)) 1.0 0.0])


(facts "Convert action to array and back"
       (action [0.0]) => {:control 0.0}
       (action [0.5]) => {:control 0.5}
       (action [2.0]) => {:control 1.0}
       (action [-2.0]) => {:control -1.0})


(facts "Check whether a run should be aborted"
       (truncate? {:t 50.0} {:timeout 100.0}) => false
       (truncate? {:t 100.0} {:timeout 100.0}) => true
       (truncate? {:t 101.0} {:timeout 100.0}) => true)


(facts "Check whether pendulum achieved target state"
       (done? {:angle 0.0 :velocity 0.0} {:target-angle 0.1 :target-velocity 0.2}) => false
       (done? {:angle (- PI 0.05) :velocity 0.0} {:target-angle 0.1 :target-velocity 0.2}) => true
       (done? {:angle (+ PI 0.05) :velocity 0.0} {:target-angle 0.1 :target-velocity 0.2}) => true
       (done? {:angle (- (- PI) 0.05) :velocity 0.0} {:target-angle 0.1 :target-velocity 0.2}) => true
       (done? {:angle (+ (- PI) 0.05) :velocity 0.0} {:target-angle 0.1 :target-velocity 0.2}) => true
       (done? {:angle PI :velocity 0.4} {:target-angle 0.1 :target-velocity 0.2}) => false
       (done? {:angle PI :velocity -0.4} {:target-angle 0.1 :target-velocity 0.2}) => false)


(facts "Reward function"
       (reward {:angle PI :velocity 0.0} {:angle-weight 3.0 :velocity-weight 5.0}) => 0.0
       (reward {:angle (+ PI 2.0) :velocity 0.0} {:angle-weight 3.0 :velocity-weight 5.0}) => -12.0
       (reward {:angle (- PI 2.0) :velocity 0.0} {:angle-weight 3.0 :velocity-weight 5.0}) => -12.0
       (reward {:angle PI :velocity 2.0} {:angle-weight 3.0 :velocity-weight 5.0}) => -20.0
       (reward {:angle PI :velocity -2.0} {:angle-weight 3.0 :velocity-weight 5.0}) => -20.0)


(def test-config
  {:length  1.0
   :friction 0.1
   :motor 1.0
   :gravitation 10.0
   :dt 1.0
   :save false
   :timeout 20.0
   :target-angle 0.1
   :target-velocity 0.2
   :angle-weight 3.0
   :velocity-weight 5.0})


(facts "Implement environment"
       (:state (->Pendulum test-config (setup 1.0))) => {:angle 1.0 :velocity 0.0 :t 0.0}
       (:state (environment-update (->Pendulum test-config (setup 0.0)) [0.5])) => {:angle 0.5 :velocity 0.5 :t 1.0}
       (environment-observation (->Pendulum test-config (setup 0.0))) => [1.0 0.0 0.0]
       (environment-done? (->Pendulum test-config (setup PI))) => true
       (environment-truncate? (->Pendulum test-config (setup 0.0))) => false
       (environment-reward (->Pendulum test-config (setup (- PI 1.0))) [0.5]) => -3.0)
