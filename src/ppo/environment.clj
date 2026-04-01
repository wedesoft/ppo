(ns ppo.environment)


(defprotocol Environment
  (environment-update [this action])
  (environment-observation [this])
  (environment-done? [this])
  (environment-truncate? [this])
  (environment-reward [this]))
