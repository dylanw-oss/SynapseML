"use strict";(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[7143],{3905:function(e,n,t){t.d(n,{Zo:function(){return d},kt:function(){return m}});var r=t(7294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function o(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function s(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?o(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var l=r.createContext({}),c=function(e){var n=r.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):s(s({},n),e)),t},d=function(e){var n=c(e.components);return r.createElement(l.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},p=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,d=i(e,["components","mdxType","originalType","parentName"]),p=c(t),m=a,_=p["".concat(l,".").concat(m)]||p[m]||u[m]||o;return t?r.createElement(_,s(s({ref:n},d),{},{components:t})):r.createElement(_,s({ref:n},d))}));function m(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var o=t.length,s=new Array(o);s[0]=p;var i={};for(var l in n)hasOwnProperty.call(n,l)&&(i[l]=n[l]);i.originalType=e,i.mdxType="string"==typeof e?e:a,s[1]=i;for(var c=2;c<o;c++)s[c]=t[c];return r.createElement.apply(null,s)}return r.createElement.apply(null,t)}p.displayName="MDXCreateElement"},1305:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return i},contentTitle:function(){return l},metadata:function(){return c},toc:function(){return d},default:function(){return p}});var r=t(3117),a=t(102),o=(t(7294),t(3905)),s=["components"],i={title:"CyberML - Anomalous Access Detection",hide_title:!0,status:"stable",url_path:"features/other/CyberML - Anomalous Access Detection",name:"CyberML - Anomalous Access Detection"},l="CyberML - Anomalous Access Detection",c={unversionedId:"features/other/CyberML - Anomalous Access Detection",id:"features/other/CyberML - Anomalous Access Detection",title:"CyberML - Anomalous Access Detection",description:"Here we demonstrate a novel CyberML model which can learn user access patterns and then automatically detect anomalous user access based on learned behavior.",source:"@site/docs/features/other/CyberML - Anomalous Access Detection.md",sourceDirName:"features/other",slug:"/features/other/CyberML - Anomalous Access Detection",permalink:"/SynapseML/docs/next/features/other/CyberML - Anomalous Access Detection",tags:[],version:"current",frontMatter:{title:"CyberML - Anomalous Access Detection",hide_title:!0,status:"stable",url_path:"features/other/CyberML - Anomalous Access Detection",name:"CyberML - Anomalous Access Detection"},sidebar:"docs",previous:{title:"ConditionalKNN - Exploring Art Across Cultures",permalink:"/SynapseML/docs/next/features/other/ConditionalKNN - Exploring Art Across Cultures"},next:{title:"DeepLearning - BiLSTM Medical Entity Extraction",permalink:"/SynapseML/docs/next/features/other/DeepLearning - BiLSTM Medical Entity Extraction"}},d=[],u={toc:d};function p(e){var n=e.components,t=(0,a.Z)(e,s);return(0,o.kt)("wrapper",(0,r.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"cyberml---anomalous-access-detection"},"CyberML - Anomalous Access Detection"),(0,o.kt)("p",null,"Here we demonstrate a novel CyberML model which can learn user access patterns and then automatically detect anomalous user access based on learned behavior.\nThe model internally uses Collaborative Filtering for Implicit Feedback as published here: ",(0,o.kt)("a",{parentName:"p",href:"http://yifanhu.net/PUB/cf.pdf"},"http://yifanhu.net/PUB/cf.pdf"),"\nand is based on Apache Spark's implementation of this: ",(0,o.kt)("a",{parentName:"p",href:"https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html"},"https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html"),"."),(0,o.kt)("p",null,"This notebook demonstrates a usage example of Anomalous Resource Access model.\nAll the model requires is a dataset in which there are 'users' which access 'resources'.\nThe model is based on Collaborative Filtering and it uses Machine Learning to learn access patterns of users and resources.\nWhen a user accesses a resource which is outside of the user's learned profile then this access recieves a high anomaly score."),(0,o.kt)("p",null,"In this notebook we provide a usage example and a synthetic dataset in which there are 3 departments:\n(1) Finance, (2) HR and (3) Engineering.\nIn the training data users access only a subset of resources from their own departments.\nTo evaluate the model we use two datasets.\nThe first contains access patterns unseen during training in which users access resources within their departments (again, resources they didn't access during training but within their department).\nThe latter contains users accessing resources from outside their department.\nWe then use the model to assign anomaly scores expecting that the first get low anomaly scores and the latter recieve high anomaly scores.\nThis is what this example demonstrates."),(0,o.kt)("p",null,"Note: the data does NOT contain information about departments, this information is implictly learned by the model by analyzing the access patterns."),(0,o.kt)("h1",{id:"create-an-azure-databricks-cluster-and-install-the-following-libs"},"Create an Azure Databricks cluster and install the following libs"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("p",{parentName:"li"},"In Cluster Libraries install from library source Maven:\nCoordinates: com.microsoft.azure:synapseml_2.12:0.9.5\nRepository: ",(0,o.kt)("a",{parentName:"p",href:"https://mmlspark.azureedge.net/maven"},"https://mmlspark.azureedge.net/maven"))),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("p",{parentName:"li"},"In Cluster Libraries install from PyPI the library called plotly"))),(0,o.kt)("h1",{id:"setup--initialization"},"Setup & Initialization"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"# this is used to produce the synthetic dataset for this test\nfrom synapse.ml.cyber.dataset import DataFactory\n\n# the access anomalies model generator\nfrom synapse.ml.cyber.anomaly.collaborative_filtering import AccessAnomaly\n\nfrom pyspark.sql import functions as f, types as t\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'import os\n\nif os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":\n    from pyspark.sql import SparkSession\n\n    spark = SparkSession.builder.getOrCreate()\n')),(0,o.kt)("h1",{id:"loadup-datasets"},"Loadup datasets"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'spark.sparkContext.setCheckpointDir("dbfs:/checkpoint_path/")\n\nfactory = DataFactory(\n    num_hr_users=25,\n    num_hr_resources=50,\n    num_fin_users=35,\n    num_fin_resources=75,\n    num_eng_users=15,\n    num_eng_resources=25,\n    single_component=True,\n)\n\ntraining_pdf = factory.create_clustered_training_data(ratio=0.4)\n\n# a tenant id is used when independant datasets originate from different tenants, in this example we set all tenants-ids to the same value\ntraining_df = spark.createDataFrame(training_pdf).withColumn("tenant_id", f.lit(0))\ningroup_df = spark.createDataFrame(\n    factory.create_clustered_intra_test_data(training_pdf)\n).withColumn("tenant_id", f.lit(0))\noutgroup_df = spark.createDataFrame(\n    factory.create_clustered_inter_test_data()\n).withColumn("tenant_id", f.lit(0))\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"training_df.show()\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"print(training_df.count())\nprint(ingroup_df.count())\nprint(outgroup_df.count())\n")),(0,o.kt)("h1",{id:"model-setup--training"},"Model setup & training"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'access_anomaly = AccessAnomaly(\n    tenantCol="tenant_id",\n    userCol="user",\n    resCol="res",\n    likelihoodCol="likelihood",\n    maxIter=1000,\n)\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"model = access_anomaly.fit(training_df)\n")),(0,o.kt)("h1",{id:"apply-model--show-result-stats"},"Apply model & show result stats"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"ingroup_scored_df = model.transform(ingroup_df)\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'ingroup_scored_df.agg(\n    f.min("anomaly_score").alias("min_anomaly_score"),\n    f.max("anomaly_score").alias("max_anomaly_score"),\n    f.mean("anomaly_score").alias("mean_anomaly_score"),\n    f.stddev("anomaly_score").alias("stddev_anomaly_score"),\n).show()\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"outgroup_scored_df = model.transform(outgroup_df)\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'outgroup_scored_df.agg(\n    f.min("anomaly_score").alias("min_anomaly_score"),\n    f.max("anomaly_score").alias("max_anomaly_score"),\n    f.mean("anomaly_score").alias("mean_anomaly_score"),\n    f.stddev("anomaly_score").alias("stddev_anomaly_score"),\n).show()\n')),(0,o.kt)("h1",{id:"examine-results"},"Examine results"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'#\n# Select a subset of results to send to Log Analytics\n#\n\nfull_res_df = outgroup_scored_df.orderBy(f.desc("anomaly_score")).cache()\n\nfrom pyspark.sql.window import Window\n\nw = Window.partitionBy("tenant_id", "user", "res").orderBy(f.desc("anomaly_score"))\n\n# select values above threshold\nresults_above_threshold = full_res_df.filter(full_res_df.anomaly_score > 1.0)\n\n# get distinct resource/user and corresponding timestamp and highest score\nresults_to_la = (\n    results_above_threshold.withColumn("index", f.row_number().over(w))\n    .orderBy(f.desc("anomaly_score"))\n    .select("tenant_id", f.col("user"), f.col("res"), "anomaly_score")\n    .where("index == 1")\n    .limit(100)\n    .cache()\n)\n\n# add a fake timestamp to the results\nresults_to_la = results_to_la.withColumn("timestamp", f.current_timestamp())\n\ndisplay(results_to_la)\n')),(0,o.kt)("h1",{id:"display-all-resource-accesses-by-users-with-highest-anomalous-score"},"Display all resource accesses by users with highest anomalous score"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"from plotly import __version__\nfrom plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, offline\n\nimport numpy as np\nimport pandas as pd\n\nprint(__version__)  # requires version >= 1.9.0\n\n# run plotly in offline mode\noffline.init_notebook_mode()\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'# Find all server accesses of users with high predicted scores\n# For display, limit to top 25 results\nresults_to_display = results_to_la.orderBy(f.desc("anomaly_score")).limit(25).cache()\ninteresting_records = full_res_df.join(results_to_display, ["user"], "left_semi")\nnon_anomalous_records = interesting_records.join(\n    results_to_display, ["user", "res"], "left_anti"\n)\n\ntop_non_anomalous_records = (\n    non_anomalous_records.groupBy("tenant_id", "user", "res")\n    .agg(\n        f.count("*").alias("count"),\n    )\n    .select(f.col("tenant_id"), f.col("user"), f.col("res"), "count")\n)\n\n# pick only a subset of non-anomalous record for UI\nw = Window.partitionBy(\n    "tenant_id",\n    "user",\n).orderBy(f.desc("count"))\n\n# pick top non-anomalous set\ntop_non_anomalous_accesses = (\n    top_non_anomalous_records.withColumn("index", f.row_number().over(w))\n    .orderBy(f.desc("count"))\n    .select("tenant_id", f.col("user"), f.col("res"), f.col("count"))\n    .where("index in (1,2,3,4,5)")\n    .limit(25)\n)\n\n# add back anomalous record\nfileShare_accesses = (\n    top_non_anomalous_accesses.select("user", "res", "count")\n    .union(results_to_display.select("user", "res", f.lit(1).alias("count")))\n    .cache()\n)\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'# get unique users and file shares\nhigh_scores_df = fileShare_accesses.toPandas()\nunique_arr = np.append(high_scores_df.user.unique(), high_scores_df.res.unique())\n\nunique_df = pd.DataFrame(data=unique_arr, columns=["name"])\nunique_df["index"] = range(0, len(unique_df.index))\n\n# create index for source & target and color for the normal accesses\nnormal_line_color = "rgba(211, 211, 211, 0.8)"\nanomolous_color = "red"\nx = (\n    pd.merge(high_scores_df, unique_df, how="left", left_on="user", right_on="name")\n    .drop(["name"], axis=1)\n    .rename(columns={"index": "userIndex"})\n)\nall_access_index_df = (\n    pd.merge(x, unique_df, how="left", left_on="res", right_on="name")\n    .drop(["name"], axis=1)\n    .rename(columns={"index": "resIndex"})\n)\nall_access_index_df["color"] = normal_line_color\n\n# results_to_display index, color and\ny = results_to_display.toPandas().drop(\n    ["tenant_id", "timestamp", "anomaly_score"], axis=1\n)\ny = (\n    pd.merge(y, unique_df, how="left", left_on="user", right_on="name")\n    .drop(["name"], axis=1)\n    .rename(columns={"index": "userIndex"})\n)\nhigh_scores_index_df = (\n    pd.merge(y, unique_df, how="left", left_on="res", right_on="name")\n    .drop(["name"], axis=1)\n    .rename(columns={"index": "resIndex"})\n)\nhigh_scores_index_df["count"] = 1\nhigh_scores_index_df["color"] = anomolous_color\n\n# substract 1 for the red entries in all_access df\nhsi_df = high_scores_index_df[["user", "res", "count"]].rename(\n    columns={"count": "hsiCount"}\n)\nall_access_updated_count_df = pd.merge(\n    all_access_index_df,\n    hsi_df,\n    how="left",\n    left_on=["user", "res"],\n    right_on=["user", "res"],\n)\nall_access_updated_count_df["count"] = np.where(\n    all_access_updated_count_df["hsiCount"] == 1,\n    all_access_updated_count_df["count"] - 1,\n    all_access_updated_count_df["count"],\n)\nall_access_updated_count_df = all_access_updated_count_df.loc[\n    all_access_updated_count_df["count"] > 0\n]\nall_access_updated_count_df = all_access_updated_count_df[\n    ["user", "res", "count", "userIndex", "resIndex", "color"]\n]\n\n# combine the two tables\nframes = [all_access_updated_count_df, high_scores_index_df]\ndisplay_df = pd.concat(frames, sort=True)\n# display_df.head()\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'data_trace = dict(\n    type="sankey",\n    domain=dict(x=[0, 1], y=[0, 1]),\n    orientation="h",\n    valueformat=".0f",\n    node=dict(\n        pad=10,\n        thickness=30,\n        line=dict(color="black", width=0),\n        label=unique_df["name"].dropna(axis=0, how="any"),\n    ),\n    link=dict(\n        source=display_df["userIndex"].dropna(axis=0, how="any"),\n        target=display_df["resIndex"].dropna(axis=0, how="any"),\n        value=display_df["count"].dropna(axis=0, how="any"),\n        color=display_df["color"].dropna(axis=0, how="any"),\n    ),\n)\n\nlayout = dict(\n    title="All resources accessed by users with highest anomalous scores",\n    height=772,\n    font=dict(size=10),\n)\n\nfig = dict(data=[data_trace], layout=layout)\n\np = plot(fig, output_type="div")\n\ndisplayHTML(p)\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"")))}p.isMDXComponent=!0}}]);