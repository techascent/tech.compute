(defproject techascent/tech.compute "4.45-1-SNAPSHOT"
  :description "Library designed to provide a generic compute abstraction to allow some level of shared implementation between a cpu, cuda, openCL, webworkers, etc."
  :url "http://github.com/tech-ascent/tech.compute"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins [[lein-tools-deps "0.4.1"]]
  :middleware [lein-tools-deps.plugin/resolve-dependencies-with-deps-edn]
  :lein-tools-deps/config {:config-files [:install :user :project]
                           :clojure-executables ["/usr/local/bin/clojure"
                                                 "/usr/bin/clojure"]}
  :java-source-paths ["java"]
  :profiles {:dev {:dependencies [[com.github.fommil.netlib/all "1.1.2" :extension "pom"]]}
             :clojure-10 {:dependencies [[org.clojure/clojure "1.10.0"]]}})
