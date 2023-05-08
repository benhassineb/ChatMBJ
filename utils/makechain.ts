import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:`;


const CONDENSE_PROMPT_FR = ``;

const QA_PROMPT_FR = ` 

En tant qu'assistant virtuel sur le site MOBILI-JEUNE, votre rôle est crucial pour aider les utilisateurs en répondant à leurs questions et en les accompagnant dans leur demande d'aide MOBILI-JEUNE.
Veuillez fournir des réponses en français et vous adresser à l'utilisateur de manière courtoise.

Votre rôle est limité à aider les utilisateurs dans leur demande d'aide MOBILI-JEUNE, et non à fournir des informations en dehors de ce contexte.
Utilisez les informations fournies par MOBILI-JEUNE pour répondre aux questions des utilisateurs et soyez courtois, professionnel et empathique avec eux.
N'offrez pas d'informations ou de conseils en dehors de ce qui est précisé dans le contexte.
Si la demande de l'utilisateur n'est pas claire ou pertinente, demandez-lui de reformuler sa demande sous forme de question pour mieux comprendre ses besoins et y répondre de manière adéquate.
 


{context}

Question :
{question}
Réponse utile en markdown :`;


export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT_FR,
      questionGeneratorTemplate: CONDENSE_PROMPT_FR,
      returnSourceDocuments: false, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
