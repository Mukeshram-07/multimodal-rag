import axios from 'axios';

const BASE_URL = import.meta.env.NEXT_PUBLIC_API_URL || '/api';

export const client = axios.create({ baseURL: BASE_URL, timeout: 120000 });

// Attach JWT on every request
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('rag_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Redirect to login on 401
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('rag_token');
      window.location.href = '/';
    }
    return Promise.reject(err);
  }
);

function handleError(err) {
  if (err.response) {
    const detail = err.response.data?.detail || err.response.data?.error || err.message;
    return { message: detail, status: err.response.status };
  }
  if (err.code === 'ECONNABORTED') return { message: 'Request timed out.', status: 0 };
  if (err.code === 'ERR_NETWORK') return { message: 'Cannot reach the API server.', status: 0 };
  return { message: err.message || 'Unknown error', status: 0 };
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function requestOtp(email) {
  try {
    const { data } = await client.post('/auth/request-otp', { email });
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function verifyOtp(email, otp) {
  try {
    const { data } = await client.post('/auth/verify-otp', { email, otp });
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function googleLogin(credential) {
  try {
    const { data } = await client.post('/auth/google', { credential });
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function getMe() {
  try {
    const { data } = await client.get('/auth/me');
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function updateMe(displayName) {
  try {
    const { data } = await client.put('/auth/me', { display_name: displayName });
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

// Legacy password flow (kept for backward compat)
export async function login(email, password) {
  try {
    const { data } = await client.post('/auth/login', { email, password });
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function signup(email, password, displayName) {
  try {
    const { data } = await client.post('/auth/signup', { email, password, display_name: displayName });
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

// ── RAG ───────────────────────────────────────────────────────────────────────

export async function ingestPDF(fileBytes, filename, collectionName) {
  const form = new FormData();
  form.append('file', new Blob([fileBytes], { type: 'application/pdf' }), filename);
  form.append('collection_name', collectionName);
  try {
    const { data } = await client.post('/ingest', form);
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function queryDocuments(query, collectionName, topK = 5, filterSource = null) {
  const payload = { query, collection_name: collectionName, top_k: topK };
  if (filterSource) payload.filter_source = filterSource;
  try {
    const { data } = await client.post('/query', payload);
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function listCollections() {
  try {
    const { data } = await client.get('/collections');
    return { data: data.collections || [] };
  } catch (err) { return { error: handleError(err) }; }
}

export async function deleteCollection(name) {
  try {
    const { data } = await client.delete(`/collections/${encodeURIComponent(name)}`);
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}

export async function checkHealth() {
  try {
    const { data } = await client.get('/health');
    return { data };
  } catch (err) { return { error: handleError(err) }; }
}
