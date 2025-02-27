// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset._
import org.apache.spark.sql._

import scala.language.existentials

/**
  * Class for handling the execution of bulk-based Tasks on workers for each partition.
  */
class BulkPartitionTask extends BasePartitionTask {
  override protected def initializeInternal(ctx: TrainingContext,
                                            shouldExecuteTraining: Boolean,
                                            isEmptyPartition: Boolean): Unit = {
    // For useSingleDataset mode, we need to add to bulk synchronization stops
    if (ctx.useSingleDatasetMode && !isEmptyPartition) {
      ctx.sharedState().incrementArrayProcessedSignal(log)
      if (!shouldExecuteTraining) ctx.sharedState().incrementDataPrepDoneSignal(log)
    }
  }

  protected def preparePartitionDataInternal(ctx: PartitionTaskContext,
                                             inputRows: Iterator[Row]): PartitionDataState = {
    // In useSingleDataset mode, we need to synchronize start of data loading
    if (ctx.shouldExecuteTraining) {
      // If in useSingleDatasetMode, we need to set a signal that helpers can start
      if (ctx.trainingCtx.useSingleDatasetMode) {
        ctx.sharedState.helperStartSignal.countDown()
        log.info(s"Initiated helper start signal on task ${ctx.taskId}, partition ${ctx.partitionId}")
      }
    } else {
      log.info(s"Waiting for helper start signal on task ${ctx.taskId}, partition ${ctx.partitionId}")
      ctx.sharedState.helperStartSignal.await()
    }

    // Store the chunked partition data in local memory and then add it to the aggregated data
    val aggregatedColumns = {
      log.info(s"Reading data on task ${ctx.taskId}, partition ${ctx.partitionId}")
      val prepAggregatedColumns: BaseChunkedColumns = getChunkedColumns(ctx, inputRows)
      log.info(s"Merging data on task ${ctx.taskId}, partition ${ctx.partitionId}")
      mergeChunksIntoAggregatedArrays(ctx, prepAggregatedColumns, isForValidation = false)
    }
    val aggregatedValidationColumns = ctx.trainingCtx.validationData.map { data =>
      val prepAggregatedColumns: BaseChunkedColumns = getChunkedColumns(ctx, data.value.toIterator)
      mergeChunksIntoAggregatedArrays(ctx, prepAggregatedColumns, isForValidation = true)
    }
    PartitionDataState(Option(aggregatedColumns), aggregatedValidationColumns)
  }

  protected def getTrainingDatasetInternal(ctx: PartitionTaskContext,
                                           dataState: PartitionDataState): LightGBMDataset = {
    generateDataset(ctx, dataState.aggregatedTrainingData.get, None)
  }

  protected def getValidationDatasetInternal(ctx: PartitionTaskContext,
                                             dataState: PartitionDataState,
                                             referenceDataset: LightGBMDataset): LightGBMDataset = {
    generateDataset(ctx, dataState.aggregatedValidationData.get, Option(referenceDataset))
  }

  protected def generateDataset(ctx: PartitionTaskContext,
                                ac: BaseAggregatedColumns,
                                referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    try {
      val datasetInner: LightGBMDataset = ac.generateDataset(ctx, referenceDataset)
      ctx.trainingCtx.columnParams.groupColumn.foreach(_ => datasetInner.addGroupColumn(ac.getGroups))
      datasetInner.setFeatureNames(ctx.trainingCtx.featureNames, ac.getNumCols)
      datasetInner
    } finally {
      ac.cleanup()
    }
  }

  private def getChunkedColumns(ctx: PartitionTaskContext, inputRows: Iterator[Row]): BaseChunkedColumns = {
    val trainingCtx = ctx.trainingCtx
    val newIterator = determineMatrixType(ctx, inputRows)
    if (!ctx.sharedState.isSparse.get) new DenseChunkedColumns(
        newIterator,
        trainingCtx.columnParams,
        trainingCtx.schema,
        trainingCtx.trainingParams.executionParams.chunkSize)
    else new SparseChunkedColumns(
        newIterator,
        trainingCtx.columnParams,
        trainingCtx.schema,
        trainingCtx.trainingParams.executionParams.chunkSize,
        trainingCtx.useSingleDatasetMode)
  }

  private def mergeChunksIntoAggregatedArrays(ctx: PartitionTaskContext,
                                              ts: BaseChunkedColumns,
                                              isForValidation: Boolean): BaseAggregatedColumns = {
    val sharedState = ctx.sharedState
    val useSingleDataset = ctx.trainingCtx.useSingleDatasetMode
    val isSparse = sharedState.isSparse.get
    val sharedDatasetState =
      if (isForValidation) sharedState.validationDatasetState
      else sharedState.datasetState

    // Determine if we are using shared single Dataset for executor, or one per partition
    val aggregatedColumns = if (!isSparse) {
      if (useSingleDataset) sharedDatasetState.denseAggregatedColumns
      else new DenseAggregatedColumns(ctx.trainingParams.executionParams.chunkSize)
    } else {
      if (useSingleDataset) sharedDatasetState.sparseAggregatedColumns
      else new SparseAggregatedColumns(ctx.trainingParams.executionParams.chunkSize)
    }

    // For the validation Dataset in useSingleDataset mode, we only want 1 copy of the data (otherwise
    // every partition appends the same broadcast-ed data). That one copy will be made by the main execution worker.
    val mergeRowsIntoDataset: Boolean =
      if (!isForValidation) true
      else !useSingleDataset || sharedState.mainExecutorWorker.get == LightGBMUtils.getTaskId

    // This will actually set the shared partitions sizes as well as load the chunked data
    if (mergeRowsIntoDataset) {
      aggregatedColumns.incrementCount(ts, ctx.partitionId)
    }

    if (useSingleDataset) {
      sharedDatasetState.arrayProcessedSignal.countDown()
      sharedDatasetState.arrayProcessedSignal.await()
    }

    // Now push the chunked data into the aggregated arrays
    if (mergeRowsIntoDataset) {
      aggregatedColumns.addRows(ts)
    }

    // As a side-effect, we release the chunked data
    ts.release()

    aggregatedColumns
  }
}
