(ns clojure-sandboxes.minimal-gnn
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :as py :refer [py. py.. py.-]]))

(require-python '[torch :as torch]
                '[torch.nn :as nn]
                '[torch.optim :as optim]
                '[numpy :as np])

;;; sudo pip3 install torch
;;; sudo pip3 install torch-geometric

;; Create a simple toy graph with more nodes for better training
(defn create-toy-graph []
  {:nodes [[1.0 0.0 0.0]  ; Node 0: class 0 (pure red)
           [0.9 0.1 0.0]  ; Node 1: class 0 (mostly red)
           [1.0 0.2 0.0]  ; Node 2: class 0 (red-ish)
           [0.0 1.0 0.0]  ; Node 3: class 1 (pure green)
           [0.1 0.9 0.0]  ; Node 4: class 1 (mostly green)
           [0.0 1.0 0.1]  ; Node 5: class 1 (green-ish)
           [0.0 0.0 1.0]  ; Node 6: class 2 (pure blue)
           [0.0 0.1 0.9]  ; Node 7: class 2 (mostly blue)
           [0.1 0.0 1.0]  ; Node 8: class 2 (blue-ish)
           [0.8 0.1 0.1]  ; Node 9: class 0 (training)
           ;; Test nodes below
           [0.95 0.05 0.0] ; Node 10: class 0 (test)
           [0.05 0.95 0.0] ; Node 11: class 1 (test)
           [0.0 0.05 0.95] ; Node 12: class 2 (test)
           [1.0 0.0 0.1]   ; Node 13: class 0 (test)
           [0.0 0.9 0.1]]  ; Node 14: class 1 (test)
   ;; Edges: connect nodes of similar classes together
   :edges [[0 0 1 1 2 3 3 4 4 5 6 6 7 7 8 9 9 0 1 2 3 4 5 6 7 8]
           [1 9 2 9 0 4 5 5 3 3 7 8 8 6 6 0 2 10 10 10 11 11 11 12 12 12]]
   :labels [0 0 0 1 1 1 2 2 2 0 0 1 2 0 1]})

;; Create PyTorch Geometric Data object
(defn create-data-object [graph geom-data torch]
  (let [numpy (py/import-module "numpy")
        x (py/call-attr-kw torch "tensor" [(py/call-attr-kw numpy "array" [(graph :nodes)] {:dtype (py/get-attr numpy "float32")})] {})
        edge-index (py/call-attr-kw torch "tensor" [(py/call-attr-kw numpy "array" [(graph :edges)] {})] {})
        y (py/call-attr-kw torch "tensor" [(py/call-attr-kw numpy "array" [(graph :labels)] {})] {})
        ;; 10 nodes for training (0-9), 5 nodes for testing (10-14)
        train-mask (py/call-attr-kw torch "tensor" [[true true true true true true true true true true false false false false false]] {:dtype (py/get-attr torch "int64")})
        test-mask (py/call-attr-kw torch "tensor" [[false false false false false false false false false false true true true true true]] {:dtype (py/get-attr torch "int64")})]
    (py/call-attr-kw geom-data "Data" [] {:x x :edge_index edge-index :y y :train_mask train-mask :test_mask test-mask})))

;; Create individual layers
(defn create-layers [num-features num-classes geom-nn nn]
  {:conv1 (py. geom-nn GCNConv num-features 16)
   :conv2 (py. geom-nn GCNConv 16 num-classes)
   :dropout (py. nn Dropout 0.5)})

;; Forward pass function
(defn forward-pass [layers data torch]
  (let [x (py/get-attr data "x")
        edge-index (py/get-attr data "edge_index")
        ;; First GCN layer
        h1 (py/call-attr (layers :conv1) "__call__" x edge-index)
        h1 (py. torch relu h1)
        h1 (py/call-attr (layers :dropout) "__call__" h1)
        ;; Second GCN layer
        h2 (py/call-attr (layers :conv2) "__call__" h1 edge-index)
        ;; Log softmax
        output (py/call-attr-kw torch "log_softmax" [h2] {:dim 1})]
    output))

;; Training function
(defn train [layers data optimizer torch nn-functional]
  (let [out (forward-pass layers data torch)
        train-mask (py. (py/get-attr data "train_mask") squeeze)
        train-indices (py. (py. train-mask nonzero) squeeze)
        loss (py/call-attr nn-functional "nll_loss"
                   (py. out index_select 0 train-indices)
                   (py/call-attr (py/get-attr data "y") "index_select" 0 train-indices))]
    (py. optimizer zero_grad)
    (py. loss backward)
    (py. optimizer step)
    (py. loss item)))

;; Test function
(defn test [layers data torch]
  (let [out (forward-pass layers data torch)
        pred (py/call-attr-kw out "argmax" [] {:dim 1})
        test-mask (py. (py/get-attr data "test_mask") squeeze)
        test-indices (py. (py. test-mask nonzero) squeeze)
        test-correct (py.
                      (py. pred index_select 0 test-indices)
                      eq
                      (py. (py/get-attr data "y") index_select 0 test-indices))
        test-acc (py.
                  (py. (py. test-correct sum) float)
                  div
                  (py. test-mask sum))]
    (py. test-acc item)))

;; Main training function
(defn train-gnn [epochs learning-rate]
  (println "Initializing Python...")
  (py/initialize!)

  (let [torch (py/import-module "torch")
        nn (py/import-module "torch.nn")
        nn-functional (py/import-module "torch.nn.functional")
        optim (py/import-module "torch.optim")
        geom-nn (py/import-module "torch_geometric.nn")
        geom-data (py/import-module "torch_geometric.data")
        itertools (py/import-module "itertools")]

    (println "Creating toy graph...")
    (let [graph (create-toy-graph)
          data (create-data-object graph geom-data torch)
          layers (create-layers 3 3 geom-nn nn) ; 3 features, 3 classes
          all-params (py. itertools chain
                                   (py. (layers :conv1) parameters)
                                   (py. (layers :conv2) parameters)
                                   (py. (layers :dropout) parameters))
          optimizer (py/call-attr-kw optim "Adam" [all-params] {:lr learning-rate})]

      (println "Starting training...")
      (dotimes [epoch epochs]
        (let [loss (train layers data optimizer torch nn-functional)
              test-acc (test layers data torch)]
          (when (zero? (mod epoch 20))
            (println (format "Epoch %d: Loss = %.4f, Test Acc = %.4f"
                            epoch loss test-acc)))))

      (let [final-acc (test layers data torch)
            predictions (py/call-attr-kw (forward-pass layers data torch) "argmax" [] {:dim 1})]
        (println (format "\nFinal Results:"))
        (println (format "Final Test Accuracy: %.4f" final-acc))
        (println "True labels:" (py. (py/get-attr data "y") tolist))
        (println "Predictions:" (py. predictions tolist))

        {:layers layers
         :data data
         :final-accuracy final-acc}))))

;; Main function
(defn -main [& args]
  (try
    (let [epochs (if (seq args) (Integer/parseInt (first args)) 300)
          learning-rate (if (> (count args) 1) (Double/parseDouble (second args)) 0.01)]
      (train-gnn epochs learning-rate))
    (catch Exception e
      (println "Error occurred:" (.getMessage e))
      (.printStackTrace e))))

(comment
  ;; Run with default parameters
  (train-gnn 100 0.01)

  ;; Run with custom parameters
  (train-gnn 50 0.005))
