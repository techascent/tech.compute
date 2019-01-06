(defproject techascent/tech.compute "3.3-SNAPSHOT"
  :description "Library designed to provide a generic compute abstraction to allow some level of shared implementation between a cpu, cuda, openCL, webworkers, etc."
  :url "http://github.com/tech-ascent/tech.compute"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[com.github.fommil.netlib/all "1.1.2" :extension "pom"]]
  :plugins [[lein-tools-deps "0.4.1"]]
  :middleware [lein-tools-deps.plugin/resolve-dependencies-with-deps-edn]
  :lein-tools-deps/config {:config-files [:install :user :project]}
  :jvm-opts ["-Xrunjdwp:server=y,transport=dt_socket,address=8000,suspend=n"])
