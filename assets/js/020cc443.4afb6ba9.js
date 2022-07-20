"use strict";(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[9480],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return y}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=r.createContext({}),c=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},p=function(e){var t=c(e.components);return r.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),m=c(n),y=a,v=m["".concat(l,".").concat(y)]||m[y]||u[y]||o;return n?r.createElement(v,s(s({ref:t},p),{},{components:n})):r.createElement(v,s({ref:t},p))}));function y(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,s=new Array(o);s[0]=m;var i={};for(var l in t)hasOwnProperty.call(t,l)&&(i[l]=t[l]);i.originalType=e,i.mdxType="string"==typeof e?e:a,s[1]=i;for(var c=2;c<o;c++)s[c]=n[c];return r.createElement.apply(null,s)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},7249:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return l},default:function(){return y},frontMatter:function(){return i},metadata:function(){return c},toc:function(){return u}});var r=n(7462),a=n(3366),o=(n(7294),n(3905)),s=["components"],i={title:"CognitiveServices - Analyze Text",hide_title:!0,status:"stable",name:"CognitiveServices - Analyze Text"},l="Cognitive Services - Analyze Text",c={unversionedId:"features/cognitive_services/CognitiveServices - Analyze Text",id:"version-0.10.0/features/cognitive_services/CognitiveServices - Analyze Text",title:"CognitiveServices - Analyze Text",description:"",source:"@site/versioned_docs/version-0.10.0/features/cognitive_services/CognitiveServices - Analyze Text.md",sourceDirName:"features/cognitive_services",slug:"/features/cognitive_services/CognitiveServices - Analyze Text",permalink:"/SynapseML/docs/features/cognitive_services/CognitiveServices - Analyze Text",tags:[],version:"0.10.0",frontMatter:{title:"CognitiveServices - Analyze Text",hide_title:!0,status:"stable",name:"CognitiveServices - Analyze Text"},sidebar:"docs",previous:{title:".NET example",permalink:"/SynapseML/docs/getting_started/dotnet_example"},next:{title:"CognitiveServices - Celebrity Quote Analysis",permalink:"/SynapseML/docs/features/cognitive_services/CognitiveServices - Celebrity Quote Analysis"}},p={},u=[],m={toc:u};function y(e){var t=e.components,n=(0,a.Z)(e,s);return(0,o.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"cognitive-services---analyze-text"},"Cognitive Services - Analyze Text"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'import os\n\nif os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":\n    from pyspark.sql import SparkSession\n\n    spark = SparkSession.builder.getOrCreate()\n    from notebookutils.mssparkutils.credentials import getSecret\n\n    os.environ["TEXT_API_KEY"] = getSecret("mmlspark-build-keys", "cognitive-api-key")\n    from notebookutils.visualization import display\n\n# put your service keys here\nkey = os.environ["TEXT_API_KEY"]\nlocation = "eastus"\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'df = spark.createDataFrame(\n    data=[\n        ["en", "Hello Seattle"],\n        ["en", "There once was a dog who lived in London and thought she was a human"],\n    ],\n    schema=["language", "text"],\n)\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"display(df)\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.cognitive import *\n\ntext_analyze = (\n    TextAnalyze()\n    .setLocation(location)\n    .setSubscriptionKey(key)\n    .setTextCol("text")\n    .setOutputCol("textAnalysis")\n    .setErrorCol("error")\n    .setLanguageCol("language")\n    # set the tasks to perform\n    .setEntityRecognitionTasks([{"parameters": {"model-version": "latest"}}])\n    .setKeyPhraseExtractionTasks([{"parameters": {"model-version": "latest"}}])\n    # Uncomment these lines to add more tasks\n    # .setEntityRecognitionPiiTasks([{"parameters": { "model-version": "latest"}}])\n    # .setEntityLinkingTasks([{"parameters": { "model-version": "latest"}}])\n    # .setSentimentAnalysisTasks([{"parameters": { "model-version": "latest"}}])\n)\n\ndf_results = text_analyze.transform(df)\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"display(df_results)\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'from pyspark.sql.functions import col\n\n# reformat and display for easier viewing\ndisplay(\n    df_results.select(\n        "language", "text", "error", col("textAnalysis").getItem(0)\n    ).select(  # we are not batching so only have a single result\n        "language", "text", "error", "textAnalysis[0].*"\n    )  # explode the Text Analytics tasks into columns\n)\n')))}y.isMDXComponent=!0}}]);