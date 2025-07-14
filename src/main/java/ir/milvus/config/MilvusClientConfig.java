package ir.milvus.config;


import io.milvus.client.MilvusServiceClient;
import io.milvus.param.ConnectParam;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MilvusClientConfig {

    @Bean
    public MilvusServiceClient milvusClient() {
        return new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost("localhost") // or move to application.yml
                        .withPort(19530)
                        .build()
        );
    }
}
