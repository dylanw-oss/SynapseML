"use strict";(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[7841],{3905:function(e,t,n){n.d(t,{Zo:function(){return m},kt:function(){return c}});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var l=a.createContext({}),p=function(e){var t=a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},m=function(e){var t=p(e.components);return a.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},d=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,l=e.parentName,m=o(e,["components","mdxType","originalType","parentName"]),d=p(n),c=r,k=d["".concat(l,".").concat(c)]||d[c]||u[c]||i;return n?a.createElement(k,s(s({ref:t},m),{},{components:n})):a.createElement(k,s({ref:t},m))}));function c(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,s=new Array(i);s[0]=d;var o={};for(var l in t)hasOwnProperty.call(t,l)&&(o[l]=t[l]);o.originalType=e,o.mdxType="string"==typeof e?e:r,s[1]=o;for(var p=2;p<i;p++)s[p]=n[p];return a.createElement.apply(null,s)}return a.createElement.apply(null,n)}d.displayName="MDXCreateElement"},2449:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return m},default:function(){return d}});var a=n(3117),r=n(102),i=(n(7294),n(3905)),s=["components"],o={title:"Regression - Auto Imports",hide_title:!0,status:"stable",name:"Regression - Auto Imports"},l=void 0,p={unversionedId:"features/regression/Regression - Auto Imports",id:"features/regression/Regression - Auto Imports",title:"Regression - Auto Imports",description:"Regression - Auto Imports",source:"@site/docs/features/regression/Regression - Auto Imports.md",sourceDirName:"features/regression",slug:"/features/regression/Regression - Auto Imports",permalink:"/SynapseML/docs/next/features/regression/Regression - Auto Imports",tags:[],version:"current",frontMatter:{title:"Regression - Auto Imports",hide_title:!0,status:"stable"},sidebar:"docs",previous:{title:"Classification - Twitter Sentiment with Vowpal Wabbit",permalink:"/SynapseML/docs/next/features/classification/Classification - Twitter Sentiment with Vowpal Wabbit"},next:{title:"Regression - Flight Delays with DataCleaning",permalink:"/SynapseML/docs/next/features/regression/Regression - Flight Delays with DataCleaning"}},m=[{value:"Regression - Auto Imports",id:"regression---auto-imports",children:[],level:2}],u={toc:m};function d(e){var t=e.components,n=(0,r.Z)(e,s);return(0,i.kt)("wrapper",(0,a.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h2",{id:"regression---auto-imports"},"Regression - Auto Imports"),(0,i.kt)("p",null,"This sample notebook is based on the Gallery ",(0,i.kt)("a",{parentName:"p",href:"https://gallery.cortanaintelligence.com/Experiment/670fbfc40c4f44438bfe72e47432ae7a"},"Sample 6: Train, Test, Evaluate\nfor Regression: Auto Imports\nDataset"),"\nfor AzureML Studio.  This experiment demonstrates how to build a regression\nmodel to predict the automobile's price.  The process includes training, testing,\nand evaluating the model on the Automobile Imports data set."),(0,i.kt)("p",null,"This sample demonstrates the use of several members of the synapseml library:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.train.html?#module-synapse.ml.train.TrainRegressor"},(0,i.kt)("inlineCode",{parentName:"a"},"TrainRegressor"))),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.stages.html?#module-synapse.ml.stages.SummarizeData"},(0,i.kt)("inlineCode",{parentName:"a"},"SummarizeData"))),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.featurize.html?#module-synapse.ml.featurize.CleanMissingData"},(0,i.kt)("inlineCode",{parentName:"a"},"CleanMissingData"))),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.train.html?#module-synapse.ml.train.ComputeModelStatistics"},(0,i.kt)("inlineCode",{parentName:"a"},"ComputeModelStatistics"))),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.automl.html?#module-synapse.ml.automl.FindBestModel"},(0,i.kt)("inlineCode",{parentName:"a"},"FindBestModel")))),(0,i.kt)("p",null,"First, import the pandas package so that we can read and parse the datafile\nusing ",(0,i.kt)("inlineCode",{parentName:"p"},"pandas.read_csv()")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'import os\n\nif os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":\n    from pyspark.sql import SparkSession\n\n    spark = SparkSession.builder.getOrCreate()\n')),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'data = spark.read.parquet(\n    "wasbs://publicwasb@mmlspark.blob.core.windows.net/AutomobilePriceRaw.parquet"\n)\n')),(0,i.kt)("p",null,"To learn more about the data that was just read into the DataFrame,\nsummarize the data using ",(0,i.kt)("inlineCode",{parentName:"p"},"SummarizeData")," and print the summary.  For each\ncolumn of the DataFrame, ",(0,i.kt)("inlineCode",{parentName:"p"},"SummarizeData")," will report the summary statistics\nin the following subcategories for each column:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Feature name"),(0,i.kt)("li",{parentName:"ul"},"Counts",(0,i.kt)("ul",{parentName:"li"},(0,i.kt)("li",{parentName:"ul"},"Count"),(0,i.kt)("li",{parentName:"ul"},"Unique Value Count"),(0,i.kt)("li",{parentName:"ul"},"Missing Value Count"))),(0,i.kt)("li",{parentName:"ul"},"Quantiles",(0,i.kt)("ul",{parentName:"li"},(0,i.kt)("li",{parentName:"ul"},"Min"),(0,i.kt)("li",{parentName:"ul"},"1st Quartile"),(0,i.kt)("li",{parentName:"ul"},"Median"),(0,i.kt)("li",{parentName:"ul"},"3rd Quartile"),(0,i.kt)("li",{parentName:"ul"},"Max"))),(0,i.kt)("li",{parentName:"ul"},"Sample Statistics",(0,i.kt)("ul",{parentName:"li"},(0,i.kt)("li",{parentName:"ul"},"Sample Variance"),(0,i.kt)("li",{parentName:"ul"},"Sample Standard Deviation"),(0,i.kt)("li",{parentName:"ul"},"Sample Skewness"),(0,i.kt)("li",{parentName:"ul"},"Sample Kurtosis"))),(0,i.kt)("li",{parentName:"ul"},"Percentiles",(0,i.kt)("ul",{parentName:"li"},(0,i.kt)("li",{parentName:"ul"},"P0.5"),(0,i.kt)("li",{parentName:"ul"},"P1"),(0,i.kt)("li",{parentName:"ul"},"P5"),(0,i.kt)("li",{parentName:"ul"},"P95"),(0,i.kt)("li",{parentName:"ul"},"P99"),(0,i.kt)("li",{parentName:"ul"},"P99.5")))),(0,i.kt)("p",null,"Note that several columns have missing values (",(0,i.kt)("inlineCode",{parentName:"p"},"normalized-losses"),", ",(0,i.kt)("inlineCode",{parentName:"p"},"bore"),",\n",(0,i.kt)("inlineCode",{parentName:"p"},"stroke"),", ",(0,i.kt)("inlineCode",{parentName:"p"},"horsepower"),", ",(0,i.kt)("inlineCode",{parentName:"p"},"peak-rpm"),", ",(0,i.kt)("inlineCode",{parentName:"p"},"price"),").  This summary can be very\nuseful during the initial phases of data discovery and characterization."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"from synapse.ml.stages import SummarizeData\n\nsummary = SummarizeData().transform(data)\nsummary.toPandas()\n")),(0,i.kt)("p",null,"Split the dataset into train and test datasets."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"# split the data into training and testing datasets\ntrain, test = data.randomSplit([0.6, 0.4], seed=123)\ntrain.limit(10).toPandas()\n")),(0,i.kt)("p",null,"Now use the ",(0,i.kt)("inlineCode",{parentName:"p"},"CleanMissingData")," API to replace the missing values in the\ndataset with something more useful or meaningful.  Specify a list of columns\nto be cleaned, and specify the corresponding output column names, which are\nnot required to be the same as the input column names. ",(0,i.kt)("inlineCode",{parentName:"p"},"CleanMissiongData"),'\noffers the options of "Mean", "Median", or "Custom" for the replacement\nvalue.  In the case of "Custom" value, the user also specifies the value to\nuse via the "customValue" parameter.  In this example, we will replace\nmissing values in numeric columns with the median value for the column.  We\nwill define the model here, then use it as a Pipeline stage when we train our\nregression models and make our predictions in the following steps.'),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.featurize import CleanMissingData\n\ncols = ["normalized-losses", "stroke", "bore", "horsepower", "peak-rpm", "price"]\ncleanModel = (\n    CleanMissingData().setCleaningMode("Median").setInputCols(cols).setOutputCols(cols)\n)\n')),(0,i.kt)("p",null,"Now we will create two Regressor models for comparison: Poisson Regression\nand Random Forest.  PySpark has several regressors implemented:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"LinearRegression")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"IsotonicRegression")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"DecisionTreeRegressor")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"RandomForestRegressor")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"GBTRegressor")," (Gradient-Boosted Trees)"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"AFTSurvivalRegression")," (Accelerated Failure Time Model Survival)"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"GeneralizedLinearRegression")," -- fit a generalized model by giving symbolic\ndescription of the linear preditor (link function) and a description of the\nerror distribution (family).  The following families are supported:",(0,i.kt)("ul",{parentName:"li"},(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"Gaussian")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"Binomial")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"Poisson")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"Gamma")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"Tweedie")," -- power link function specified through ",(0,i.kt)("inlineCode",{parentName:"li"},"linkPower"),"\nRefer to the\n",(0,i.kt)("a",{parentName:"li",href:"http://spark.apache.org/docs/latest/api/python/"},"Pyspark API Documentation"),"\nfor more details.")))),(0,i.kt)("p",null,(0,i.kt)("inlineCode",{parentName:"p"},"TrainRegressor")," creates a model based on the regressor and other parameters\nthat are supplied to it, then trains data on the model."),(0,i.kt)("p",null,"In this next step, Create a Poisson Regression model using the\n",(0,i.kt)("inlineCode",{parentName:"p"},"GeneralizedLinearRegressor")," API from Spark and create a Pipeline using the\n",(0,i.kt)("inlineCode",{parentName:"p"},"CleanMissingData")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"TrainRegressor")," as pipeline stages to create and\ntrain the model.  Note that because ",(0,i.kt)("inlineCode",{parentName:"p"},"TrainRegressor")," expects a ",(0,i.kt)("inlineCode",{parentName:"p"},"labelCol")," to\nbe set, there is no need to set ",(0,i.kt)("inlineCode",{parentName:"p"},"linkPredictionCol")," when setting up the\n",(0,i.kt)("inlineCode",{parentName:"p"},"GeneralizedLinearRegressor"),".  Fitting the pipe on the training dataset will\ntrain the model.  Applying the ",(0,i.kt)("inlineCode",{parentName:"p"},"transform()")," of the pipe to the test dataset\ncreates the predictions."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'# train Poisson Regression Model\nfrom pyspark.ml.regression import GeneralizedLinearRegression\nfrom pyspark.ml import Pipeline\nfrom synapse.ml.train import TrainRegressor\n\nglr = GeneralizedLinearRegression(family="poisson", link="log")\npoissonModel = TrainRegressor().setModel(glr).setLabelCol("price").setNumFeatures(256)\npoissonPipe = Pipeline(stages=[cleanModel, poissonModel]).fit(train)\npoissonPrediction = poissonPipe.transform(test)\n')),(0,i.kt)("p",null,"Next, repeat these steps to create a Random Forest Regression model using the\n",(0,i.kt)("inlineCode",{parentName:"p"},"RandomRorestRegressor")," API from Spark."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'# train Random Forest regression on the same training data:\nfrom pyspark.ml.regression import RandomForestRegressor\n\nrfr = RandomForestRegressor(maxDepth=30, maxBins=128, numTrees=8, minInstancesPerNode=1)\nrandomForestModel = TrainRegressor(model=rfr, labelCol="price", numFeatures=256).fit(\n    train\n)\nrandomForestPipe = Pipeline(stages=[cleanModel, randomForestModel]).fit(train)\nrandomForestPrediction = randomForestPipe.transform(test)\n')),(0,i.kt)("p",null,"After the models have been trained and scored, compute some basic statistics\nto evaluate the predictions.  The following statistics are calculated for\nregression models to evaluate:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Mean squared error"),(0,i.kt)("li",{parentName:"ul"},"Root mean squared error"),(0,i.kt)("li",{parentName:"ul"},"R^2"),(0,i.kt)("li",{parentName:"ul"},"Mean absolute error")),(0,i.kt)("p",null,"Use the ",(0,i.kt)("inlineCode",{parentName:"p"},"ComputeModelStatistics")," API to compute basic statistics for\nthe Poisson and the Random Forest models."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.train import ComputeModelStatistics\n\npoissonMetrics = ComputeModelStatistics().transform(poissonPrediction)\nprint("Poisson Metrics")\npoissonMetrics.toPandas()\n')),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'randomForestMetrics = ComputeModelStatistics().transform(randomForestPrediction)\nprint("Random Forest Metrics")\nrandomForestMetrics.toPandas()\n')),(0,i.kt)("p",null,"We can also compute per instance statistics for ",(0,i.kt)("inlineCode",{parentName:"p"},"poissonPrediction"),":"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.train import ComputePerInstanceStatistics\n\n\ndef demonstrateEvalPerInstance(pred):\n    return (\n        ComputePerInstanceStatistics()\n        .transform(pred)\n        .select("price", "prediction", "L1_loss", "L2_loss")\n        .limit(10)\n        .toPandas()\n    )\n\n\ndemonstrateEvalPerInstance(poissonPrediction)\n')),(0,i.kt)("p",null,"and with ",(0,i.kt)("inlineCode",{parentName:"p"},"randomForestPrediction"),":"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"demonstrateEvalPerInstance(randomForestPrediction)\n")))}d.isMDXComponent=!0}}]);