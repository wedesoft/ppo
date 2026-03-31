(ns ppo.xor
    (:gen-class)
    (:require [libpython-clj2.require :refer [require-python]]
              [libpython-clj2.python :refer [py.] :as py]))

(require-python '[torch :as torch]
                '[torch.nn :as nn]
                '[torch.nn.functional :as F]
                '[torch.optim :as optim])


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


(defn -main [& _args]
  (let [model         (XORNet)
        data          (torch/tensor [[0 0] [0 1] [1 0] [1 1]] :dtype torch/float32)
        label         (torch/tensor [[0] [1] [1] [0]] :dtype torch/float32)
        criterion     (nn/BCELoss)
        epochs        10000
        learning-rate 0.1
        optimizer     (optim/Adam (py. model "parameters") :lr learning-rate :weight_decay 0.001)]

    ; Train model
    (py. model train)
    (doseq [epoch (range epochs)]
           (py. optimizer zero_grad)
           (let [prediction (py. model __call__ data)
                 loss       (py. criterion __call__ prediction label)]
             (py. loss backward)
             (py. optimizer step)
             (when (= (mod (inc epoch) 1000) 0)
               (println (str "epoch: " (inc epoch) " loss: " (py. loss item))))))

    ; Run inference
    (py. model eval)
    (let [no-grad (torch/no_grad)]
      (try
        (py. no-grad __enter__)
        (doseq [input data]
               (println input "->" (py. model __call__ input)))
        (finally
          (py. no-grad __exit__ nil nil nil))))

    (System/exit 0)))
