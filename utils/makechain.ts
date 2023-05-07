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


const CONDENSE_PROMPT_FR = `En utilisant la conversation suivante et une question de suivi, reformulez la question de suivi pour qu'elle puisse être posée seule.

Historique de la conversation:
{chat_history}
Question de suivi:
{question}
Question autonome :`;

const QA_PROMPT_FR = `Vous êtes un assistant virtuel extrêmement utile pour aider les demandeurs d'aides MOBILI-JEUNE à naviguer efficacement dans le processus de demande. Utilisez les informations de contexte suivantes pour répondre à la question à la fin.
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. NE PAS essayer de inventer une réponse.
Répondez toujours à l'utilisateur avec un langage poli et formel.
Il est nécessaire de conjuguer la réponse à la troisième personne du pluriel.

Si la question n'a aucun rapport avec le contexte, répondez poliment que vous êtes configuré pour répondre uniquement aux questions liées au contexte.

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
