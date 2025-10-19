(ns clojure-sandboxes.test-imports
  (:require [libpython-clj2.python :as py :refer [py. py.. py.-]]))

(defn -main [& args]
  (try
    (println "Testing PyTorch imports...")
    (py/initialize!)
    (let [torch (py/import-module "torch")]
      (println "PyTorch version:" (py/get-attr torch "__version__"))
      (println "Test successful!"))
    (catch Exception e
      (println "Error occurred:" (.getMessage e))
      (.printStackTrace e))))