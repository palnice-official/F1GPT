import { DataAPIClient, Db } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { GoogleGenAI } from "@google/genai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import { launch } from "puppeteer";
type SimilarityMetric = "dot_product" | "cosine" | "euclidean";

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  GEMINI_API_KEY,
} = process.env;

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

const f1Data = ["https://en.wikipedia.org/wiki/Formula_One"];
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const creatCollection = async (
  SimilarityMetric: SimilarityMetric = "dot_product"
) => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: {
      dimension: 768,
      metric: SimilarityMetric,
    },
  });
  console.log(res);
};

const loadSampleData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);
  for await (const url of f1Data) {
    const content = await scrapePage(url);

    const chunks = await splitter.splitText(content);
    for await (const chunk of chunks) {
      const embedding = await ai.models.embedContent({
        model: "gemini-embedding-001",
        contents: chunk,
      });

      const vector = embedding[0]

      const res = await collection.insertOne({
        $vector: vector,
        text: chunk
      })
      console.log(res)
    }
  }
};

const scrapePage = async (url: string) => {
    const loader = new PuppeteerWebBaseLoader(url, {
        launchOptions: {
            headless: true,
        },
        gotoOptions: {
            waitUntil: "domcontentloaded"
        },
        evaluate: async (page) => { 
            const result = await page.evaluate(() => document.body.innerHTML);
            return result;
        }
    });

    return (await loader.scrape())?.replace(/<[^>]*>?/gm, '');
};

creatCollection().then(() => loadSampleData())