(ns ppo.xor
    (:gen-class)
    (:require [libpython-clj2.require :refer [require-python]]
              [libpython-clj2.python :as py]))

(require-python '[torch :as torch]
                '[torch.nn :as nn])
