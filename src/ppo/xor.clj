(ns ppo.xor
    (:gen-class)
    (:import [org.encog.neural.networks BasicNetwork]
             [org.encog.neural.networks.layers BasicLayer]
             [org.encog.engine.network.activation ActivationSigmoid]
             [org.encog.ml.data.basic BasicMLDataSet]
             [org.encog.neural.networks.training.propagation.resilient ResilientPropagation]))

(def xor-input  (into-array (map double-array [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]])))
(def xor-ideal  (into-array (map double-array [[0.0] [1.0] [1.0] [0.0]])))

(defn -main
  [& _args]
  (let [training-set (BasicMLDataSet. xor-input xor-ideal)
        network (BasicNetwork.)]

    (doto network
      (.addLayer (BasicLayer. nil true 2))                      ;; Input: 2 nodes
      (.addLayer (BasicLayer. (ActivationSigmoid.) true 3))     ;; Hidden: 3 nodes
      (.addLayer (BasicLayer. (ActivationSigmoid.) false 1)))   ;; Output: 1 node 
    (.finalizeStructure (.getStructure network))
    (.reset network)

    (let [train (ResilientPropagation. network training-set)]
      (loop [epoch 1]
        (.iteration train)
        (let [error (.getError train)]
          (println (format "Epoch #%d Error: %.5f" epoch error))
          (if (and (> error 0.001) (< epoch 1000)) ;; Stop if error is low or 1000 epochs hit
            (recur (inc epoch))
            (println "Training complete."))))

    (println "\nInference Results:")
    (doseq [pair training-set]
      (let [input (.getInput pair)
            output (.compute network input)]
        (println (format "Input: [%.1f, %.1f] -> Predicted: %.4f (Actual: %.1f)"
                         (.getData input 0)
                         (.getData input 1)
                         (.getData output 0)
                         (.getData (.getIdeal pair) 0))))))))
