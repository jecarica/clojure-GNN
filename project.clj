(defproject clojure-sandboxes "0.1.0-SNAPSHOT"
  :description "Clojure GNN toy example"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                [clj-python/libpython-clj "2.025"]]
  :main ^:skip-aot clojure-sandboxes.minimal-gnn
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all
                       :jvm-opts ["-Dclojure.compiler.direct-linking=true"]}})
