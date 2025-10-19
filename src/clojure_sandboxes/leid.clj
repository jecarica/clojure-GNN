(ns clojure-sandboxes.leid
  (:require [libpython-clj2.require :refer [require-python]]
              [libpython-clj2.python :as py :refer [py. py.. py.-]]))

;;;sudo pip3 install leidenalg

;;; you also need to make sure igraph is working and installed too (see igraph.clj)

;;; What is leidenalg? https://github.com/vtraag/leidenalg
;; Implementation of the Leiden algorithm for various quality functions to be used with igraph in Python.
;;; sudo pip3 install pycairo

(require-python '[igraph :as ig])
(require-python '[leidenalg :as la]
'[torch :as torch]
 '[torch.cuda :as cuda]
 '[torch.onnx :as onnx]
 '[torch.nn :as nn :refer [Conv2d Dropout2d Linear]]
 '[torch.optim :as optim]
 '[torch.utils.data :as tud]
 '[torch.nn.functional :as F]
 '[torchvision.datasets :as datasets]
 '[torchvision.transforms :as transforms]
 '[torch.optim.lr_scheduler :as lr_scheduler]
 )

;;https://leidenalg.readthedocs.io/en/latest/intro.html

;;Let us then look at one of the most famous examples of network science: the Zachary karate club (it even has a prize named after it):
(def G (py. (ig/Graph) Famous "Zachary"))

;;;Now detecting communities with modularity is straightforward


(def partition (la/find_partition G la/ModularityVertexPartition))

;;; plotting results

(def plot (ig/plot partition))

;;; save the plot png

(py. plot save "zach.png")


(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (py. plot save "zach.png"))