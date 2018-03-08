(defproject techascent/tech.compute "0.4.1-SNAPSHOT"
  :description "Library designed to provide a generic compute abstraction to allow some level of shared implementation between a cpu, cuda, openCL, webworkers, etc."
  :url "http://github.com/tech-ascent/tech.compute"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.javacpp-datatype "0.3.2"]
                 [thinktopic/think.parallel "0.3.8"]
                 [org.clojure/math.combinatorics "0.1.4"]
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]])
