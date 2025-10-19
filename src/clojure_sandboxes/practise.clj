(ns clojure-sandboxes.practise)


(defn greedy-recur
  ""
  ([denominations amount]
   {:pre [(not (empty? denominations))
          (not (neg? amount))]}
   (greedy-recur (apply sorted-set denominations) amount (zipmap denominations (repeat 0))))
  ([denominations amount res-map]
   (if (> amount 0)
     (let [;d (apply min-key #(quot amount %) denominations)
           ;d (apply max denominations)
           d (last denominations)
           d-num (quot amount d)
           new-amount (- amount (* d d-num))]
       (recur (disj denominations d) new-amount (assoc res-map d d-num)))
     res-map)))

(defn greedy-reduce
  ""
  [denominations amount]
  (reduce (fn [res-map d]
            (let [new-amount (reduce (fn [n-a [k v]] (- n-a (* k v))) amount res-map)]
              (assoc res-map d (quot new-amount d))))
          (zipmap denominations (repeat 0))
          (reverse (apply sorted-set denominations))))
