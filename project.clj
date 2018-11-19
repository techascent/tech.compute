(defproject techascent/tech.compute "2.0"
  :description "Library designed to provide a generic compute abstraction to allow some level of shared implementation between a cpu, cuda, openCL, webworkers, etc."
  :url "http://github.com/tech-ascent/tech.compute"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.javacpp-datatype "2.2"]
                 [techascent/tech.parallel "1.0"]
                 ;;Backup for if we can't load system blas.  We then do array-backed
                 ;;nio buffers and use the netlib blas (most likely a lot slower).
                 [com.github.fommil.netlib/all "1.1.2" :extension "pom"]])
