import { input } from '@inquirer/prompts';
import fetch from 'node-fetch';
import readline from 'readline';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OllamaEmbeddings, ChatOllama } from '@langchain/ollama';
import { Document } from '@langchain/core/documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';

async function main() {
  // 1. Prompt for markdown file URL
  const url = await input({ message: 'Enter the URL to your markdown file:' });

  // 2. Download markdown file
  console.log('Downloading markdown...');
  const response = await fetch(url);
  if (!response.ok) {
    console.error('Failed to download file:', response.statusText);
    process.exit(1);
  }
  const markdown = await response.text();

  // 3. Chunk markdown
  console.log('Chunking markdown...');
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ['\n\n', '\n', ' ', ''],
  });
  // Wrap in Document for LangChain compatibility
  const docs = [new Document({ pageContent: markdown, metadata: { source: url } })];
  const chunks = await splitter.splitDocuments(docs);

  // 4. Embed and index with MemoryVectorStore
  console.log('Embedding and indexing...');
  const embeddings = new OllamaEmbeddings({
    model: 'granite3.3:2b',
    baseUrl: 'http://localhost:11434',
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

  // 5. Set up LLM and prompt template
  const llm = new ChatOllama({
    model: 'granite3.3:2b',
    temperature: 0.1,
    baseUrl: 'http://localhost:11434',
  });
  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      'system',
      `You are an expert documentation assistant. Use the following context to answer questions about the documentation accurately and helpfully.
Context: {context}
Guidelines:
- Provide accurate information based only on the provided context
- Include relevant code examples when available
- Mention the source document when possible
- If information is not in the context, clearly state that`,
    ],
    ['human', '{question}'],
  ]);

  // 6. CLI chat loop
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> ',
  });

  console.log('\nReady! Ask your questions about the document. Type "exit" to quit.');
  rl.prompt();

  rl.on('line', async (line) => {
    if (line.trim().toLowerCase() === 'exit') {
      rl.close();
      return;
    }
    try {
      const retriever = vectorStore.asRetriever({ k: 5 });
      const relevantDocs = await retriever.getRelevantDocuments(line.trim());
      const context = relevantDocs.map(doc => doc.pageContent).join('\n\n');
      // FIX: Use formatMessages to get chat messages for the LLM
      const promptMessages = await promptTemplate.formatMessages({
        context: context,
        question: line.trim(),
      });
      const response = await llm.invoke(promptMessages);
      console.log(`Answer: ${response.content}\n`);
    } catch (err) {
      console.error('Error:', err.message);
    }
    rl.prompt();
  });
}

main();


