package ir.milvus.service;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.milvus.client.MilvusServiceClient;
import io.milvus.common.clientenum.ConsistencyLevelEnum;
import io.milvus.grpc.DataType;
import io.milvus.grpc.MutationResult;
import io.milvus.grpc.SearchResults;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.index.CreateIndexParam;
import io.milvus.response.SearchResultsWrapper;
import ir.utils.CommonUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.nio.ByteBuffer;
import java.util.*;

@Service
@RequiredArgsConstructor
public class MilvusVectorService {

    private static final String COLLECTION_NAME = "java_sdk_example_float16_vector_v1";
    private static final String ID_FIELD = "id";
    private static final String VECTOR_FIELD = "vector";
    private static final Integer VECTOR_DIM = 128;

    private final MilvusServiceClient milvusClient;

    public void runTest(boolean bfloat16) {
        createCollection(bfloat16);

        // your insert/search/query logic goes here...
        int batchRowCount = 5000;
        List<List<Float>> originVectors = CommonUtils.generateFloatVectors(VECTOR_DIM, batchRowCount);
        List<ByteBuffer> encodedVectors = CommonUtils.encodeFloat16Vectors(originVectors, bfloat16);

        List<Long> ids = new ArrayList<>();
        for (long i = 0L; i < batchRowCount; ++i) {
            ids.add(i);
        }
        List<InsertParam.Field> fieldsInsert = new ArrayList<>();
        fieldsInsert.add(new InsertParam.Field(ID_FIELD, ids));
        fieldsInsert.add(new InsertParam.Field(VECTOR_FIELD, encodedVectors));

        R<MutationResult> insertR = insert(COLLECTION_NAME, fieldsInsert);

        CommonUtils.handleResponseStatus(insertR);

        System.out.println("Run logic with bfloat16 = " + bfloat16);

///   /////////search////////////////////////////////////////////////////////////////////////////////////////////////////////

        for (int i = 0; i < 10; i++) {
            Random ran = new Random();
            int k = ran.nextInt(batchRowCount);
            ByteBuffer targetVector = encodedVectors.get(k);
            SearchParam.Builder builder = SearchParam.newBuilder()
                    .withCollectionName(COLLECTION_NAME)
                    .withMetricType(MetricType.COSINE)
                    .withTopK(3)
                    .withVectorFieldName(VECTOR_FIELD)
                    .addOutField(VECTOR_FIELD)
                    .withParams("{\"nprobe\":128}");
            if (bfloat16) {
                builder.withBFloat16Vectors(Collections.singletonList(targetVector));
            } else {
                builder.withFloat16Vectors(Collections.singletonList(targetVector));
            }
            R<SearchResults> searchRet = milvusClient.search(builder.build());
            CommonUtils.handleResponseStatus(searchRet);
            // The search() allows multiple target vectors to search in a batch.
            // Here we only input one vector to search, get the result of No.0 vector to check
            SearchResultsWrapper resultsWrapper = new SearchResultsWrapper(searchRet.getData().getResults());
            List<SearchResultsWrapper.IDScore> scores = resultsWrapper.getIDScore(0);
            System.out.printf("The result of No.%d target vector:\n", i);

            SearchResultsWrapper.IDScore firstScore = scores.getFirst();
            if (firstScore.getLongID() != k) {
                throw new RuntimeException(String.format("The top1 ID %d is not equal to target vector's ID %d",
                        firstScore.getLongID(), k));
            }

            ByteBuffer outputBuf = (ByteBuffer)firstScore.get(VECTOR_FIELD);
            if (!outputBuf.equals(targetVector)) {
                throw new RuntimeException(String.format("The output vector is not equal to target vector: ID %d", k));
            }

            List<Float> outputVector = CommonUtils.decodeFloat16Vector(outputBuf, bfloat16);
            List<Float> originVector = originVectors.get(k);
            for (int j = 0; j < outputVector.size(); j++) {
                if (!isFloat16Eauql(outputVector.get(j), originVector.get(j), bfloat16)) {
                    throw new RuntimeException(String.format("The output vector is not equal to original vector: ID %d", k));
                }
            }
            System.out.println("\nTarget vector: " + originVector);
            System.out.println("Top0 result: " + firstScore);
            System.out.println("Top0 result vector: " + outputVector);
        }
        System.out.println("Search result is correct");

        dropCollection();
    }

    private void createCollection(boolean bfloat16) {
        DataType dataType = bfloat16 ? DataType.BFloat16Vector : DataType.Float16Vector;

        if (milvusClient.hasCollection(HasCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .build()).getData()) {
            dropCollection();
        }

        List<FieldType> fields = Arrays.asList(
                FieldType.newBuilder().withName(ID_FIELD).withDataType(DataType.Int64).withPrimaryKey(true).withAutoID(false).build(),
                FieldType.newBuilder().withName(VECTOR_FIELD).withDataType(dataType).withDimension(VECTOR_DIM).build()
        );

        milvusClient.createCollection(CreateCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withFieldTypes(fields)
                .withConsistencyLevel(ConsistencyLevelEnum.STRONG)
                .build());

        milvusClient.createIndex(CreateIndexParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withFieldName(VECTOR_FIELD)
                .withIndexType(IndexType.IVF_FLAT)
                .withMetricType(MetricType.COSINE)
                .withExtraParam("{\"nlist\":128}")
                .build());

        milvusClient.loadCollection(LoadCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .build());
    }

    public R<MutationResult> insert(String collectionName, List<InsertParam.Field> fields) {
        if (fields == null || fields.isEmpty()) {
            throw new IllegalArgumentException("Rows can't be null or empty");
        }

        return milvusClient.insert(InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(fields)
                .build());
    }

    private static boolean isFloat16Eauql(Float a, Float b, boolean bfloat16) {
        if (bfloat16) {
            return Math.abs(a - b) <= 0.01f;
        } else {
            return Math.abs(a - b) <= 0.001f;
        }
    }

    private void dropCollection() {
        milvusClient.dropCollection(DropCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .build());
    }
}