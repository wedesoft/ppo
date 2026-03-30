(ns ppo.xor
    (:gen-class)
    (:require [libpython-clj2.require :refer [require-python]]
              [libpython-clj2.python :refer [py.] :as py]))

(require-python '[torch :as torch]
                '[torch.nn :as nn]
                '[torch.nn.functional :as F])


(def XORNet
  (py/create-class
    "XORNet" [nn/Module]
    {"__init__"
     (py/make-instance-fn
       (fn [self]
           (py. nn/Module __init__ self)
           (py/set-attrs!
             self
             {"fc1" (nn/Linear 2 5)
              "fc2" (nn/Linear 5 1)
              "sigmoid" (nn/Sigmoid)})
           nil))
     "forward"
     (py/make-instance-fn
       (fn [self x]
           (let [x (py. self fc1 x)
                 x (F/relu x)
                 x (py. self fc2 x)
                 x (py. self sigmoid x)]
             x)))}))


(def model (XORNet))
(py. model __call__ (torch/tensor [1.0 1.0] :dtype torch/float32))
