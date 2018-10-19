(defproject techascent/tech.compute "0.7.1"
  :description "Library designed to provide a generic compute abstraction to allow some level of shared implementation between a cpu, cuda, openCL, webworkers, etc."
  :url "http://github.com/tech-ascent/tech.compute"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.datatype "0.5.15"]
                 [techascent/tech.parallel "1.0"]
                 [techascent/tech.resource "2.0"]
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]])
