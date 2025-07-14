package ir.milvus;

import ir.milvus.service.MilvusVectorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MilvusExampleApplication implements CommandLineRunner {

    @Autowired
    private MilvusVectorService milvusVectorService;


    public static void main(String[] args) {
        SpringApplication.run(MilvusExampleApplication.class, args);
    }


    @Override
    public void run(String... args) {
        milvusVectorService.runTest(true);
        milvusVectorService.runTest(false);
    }

}
