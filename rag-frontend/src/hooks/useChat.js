import { useState, useCallback } from 'react';
import { queryDocuments } from '../services/api';

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendQuery = useCallback(async (query, collectionName, topK, filterSource) => {
    const userMsg = { id: Date.now(), role: 'user', content: query };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    const result = await queryDocuments(query, collectionName, topK, filterSource || null);
    setLoading(false);

    if (result.error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'error',
        content: result.error.message,
      }]);
    } else {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: result.data.answer,
        citations: result.data.citations || [],
      }]);
    }
  }, []);

  const clearChat = useCallback(() => setMessages([]), []);

  return { messages, loading, sendQuery, clearChat };
}
