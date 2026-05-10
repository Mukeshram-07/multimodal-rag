import { useState, useEffect, useCallback } from 'react';
import { listCollections, deleteCollection } from '../services/api';

export function useCollections() {
  const [collections, setCollections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    const result = await listCollections();
    setLoading(false);
    if (result.error) {
      setError(result.error.message);
    } else {
      setCollections(result.data);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const remove = useCallback(async (name) => {
    const result = await deleteCollection(name);
    if (!result.error) {
      setCollections(prev => prev.filter(c => c !== name));
    }
    return result;
  }, []);

  return { collections, loading, error, refresh, remove };
}
