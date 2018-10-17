(ns tech.compute.registry
  "Place to store global information about the drivers available to the compute
  subystem."
  (:require [tech.compute.driver :as drv]))


(def ^:dynamic *registered-drivers* (atom {}))


(defn- find-driver
  [driver-name]
  (get @*registered-drivers* driver-name))


(defn driver
  [driver-name]
  (if-let [retval (find-driver driver-name)]
    retval
    (throw (ex-info (format "Failed to find driver.  Perhaps a require is missing?" )
                    {:driver-name driver-name}))))


(defn register-driver
  [driver]
  (swap! *registered-drivers* assoc (drv/driver-name driver) driver))


(defn driver-names
  []
  (keys @*registered-drivers*))


(defmacro current-ns->keyword
  "Use this to name your driver."
  []
  `(keyword (str *ns*)))
