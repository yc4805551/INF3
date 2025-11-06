import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import mammoth from 'mammoth';
import { marked } from 'marked';

// FIX: Modified debounce to return a function with a `clearTimeout` method to cancel pending calls.
const debounce = <F extends (...args: any[]) => any>(func: F, waitFor: number) => {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  const debounced = (...args: Parameters<F>): void => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => func(...args), waitFor);
  };

  debounced.clearTimeout = () => {
    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }
  };
  return debounced;
};

// Define structures for the analysis modes
interface NoteAnalysis {
  organizedText: string;
  userThoughts: string;
}

// Updated AuditIssue to support dynamic checklists and explanations
interface AuditIssue {
  problematicText: string;
  suggestion: string;
  checklistItem: string; // The rule from the checklist that was violated
  explanation: string; // Explanation of the issue
}

interface WritingSuggestion {
    originalText: string;
    revisedText: string;
    explanation: string;
}

interface Source {
  source_file: string;
  content_chunk: string;
  score: number;
}

// New interface for strongly typing roaming results
interface RoamingResultItem {
  source: string;
  relevantText: string;
  conclusion: string;
}

type NoteChatMessage = {
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
  sources?: Source[];
  isComplete?: boolean;
};

type ModelProvider = 'gemini' | 'openai' | 'deepseek' | 'ali';
type ChatMessage = {
  role: 'user' | 'model';
  parts: { text: string }[];
  resultType?: 'notes';
  resultData?: NoteAnalysis;
};

// NEW: Helper to get model endpoint/name from environment variables
const getModelConfig = (provider: ModelProvider) => {
    switch (provider) {
        case 'openai':
            return {
                endpoint: process.env.OPENAI_ENDPOINT!,
                model: process.env.OPENAI_MODEL!,
                providerName: 'OpenAI'
            };
        case 'deepseek':
            return {
                endpoint: process.env.DEEPSEEK_ENDPOINT!,
                model: process.env.DEEPSEEK_MODEL!,
                providerName: 'DeepSeek'
            };
        case 'ali':
            return {
                endpoint: process.env.ALI_ENDPOINT!,
                model: process.env.ALI_MODEL!,
                providerName: 'Ali'
            };
        default:
            // This case should not be hit due to checks in calling functions
            return { endpoint: '', model: '', providerName: provider };
    }
};


// State for multi-model audit results
// FIX: Defined an interface for a single audit result to provide strong typing
// for what was previously an anonymous object structure, resolving 'unknown' type errors.
interface AuditResult {
    issues: AuditIssue[];
    error?: string;
    rawResponse?: string;
}

type AuditResults = {
    [key in ModelProvider]?: AuditResult
};

const callGenerativeAi = async (provider: ModelProvider, systemInstruction: string, userPrompt: string, jsonResponse: boolean, apiKeys: {[key in ModelProvider]?: string}, mode: 'notes' | 'audit' | 'roaming' | 'writing' | null, history: ChatMessage[] = [], responseSchema?: any) => {
  const retries = 2; // 1 initial attempt + 2 retries

  for (let i = 0; i <= retries; i++) {
    try {
      let responseText: string;

      if (provider === 'gemini') {
        const geminiApiKey = process.env.API_KEY;
        if (!geminiApiKey || geminiApiKey.trim() === '' || geminiApiKey === 'undefined') {
            throw new Error("Gemini API key 未配置。请在您的 .env 文件中设置 VITE_GEMINI_API_KEY 变量。如果部署在 GitHub Pages, 请确保已在仓库的 Secrets 中正确配置 VITE_GEMINI_API_KEY。");
        }
        const ai = new GoogleGenAI({ apiKey: geminiApiKey });
        const config: any = { systemInstruction };

        if (jsonResponse) {
          config.responseMimeType = "application/json";
          if (responseSchema) {
            config.responseSchema = responseSchema;
          } else if (mode === 'notes') {
            config.responseSchema = { type: Type.OBJECT, properties: { organizedText: { type: Type.STRING }, userThoughts: { type: Type.STRING } } };
          } else if (mode === 'roaming') {
            config.responseSchema = { type: Type.OBJECT, properties: { conclusion: { type: Type.STRING } } };
          } else if (mode === 'audit') {
            config.responseSchema = {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        problematicText: { type: Type.STRING },
                        suggestion: { type: Type.STRING },
                        checklistItem: { type: Type.STRING },
                        explanation: { type: Type.STRING },
                    }
                }
            };
          } else if (mode === 'writing') {
             config.responseSchema = {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        originalText: { type: Type.STRING },
                        revisedText: { type: Type.STRING },
                        explanation: { type: Type.STRING },
                    }
                }
            };
          }
        }
        
        const fullContents = [...history.filter(h => h.parts && h.parts.length > 0), { role: 'user', parts: [{ text: userPrompt }] }];
        const contentsForApi = fullContents.map(({ role, parts }) => ({ role, parts }));
        
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: contentsForApi,
          config: config
        });
        responseText = response.text ?? '';

      } else if (provider === 'openai' || provider === 'deepseek' || provider === 'ali') {
        const { endpoint, model, providerName } = getModelConfig(provider);
        if (!endpoint || !model) {
            throw new Error(`Configuration for provider ${provider} is missing. Please check your .env file.`);
        }
        
        const transformedHistory = history.filter(h => h.parts && h.parts.length > 0).map(msg => ({
          role: msg.role === 'model' ? 'assistant' : 'user',
          content: msg.parts[0].text
        }));

        const body: any = {
            model,
            messages: [{ role: 'system', content: systemInstruction }, ...transformedHistory, { role: 'user', content: userPrompt }],
            max_tokens: 4096
        };
        if (jsonResponse) {
            body.response_format = { type: 'json_object' };
        }
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKeys[provider]}`
            },
            body: JSON.stringify(body)
        });
        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = errorText;

            if (response.status >= 500 && response.status < 600) {
                 errorMessage = `代理服务器连接失败 (状态码: ${response.status})。这通常是由于网络问题、VPN配置错误或目标API服务暂时不可用导致的。请检查您的网络连接并重试。`;
            } else if (errorText) {
                try {
                    const errorJson = JSON.parse(errorText);
                    errorMessage = errorJson.error?.message || errorMessage;
                } catch (e) { /* Ignore if parsing fails */ }
            }
            throw new Error(`${providerName} API 错误: ${response.status} ${response.statusText} - ${errorMessage}`);
        }
        const responseBodyText = await response.text();
        if (!responseBodyText) {
          throw new Error(`${providerName} returned an empty response.`);
        }
        try {
            const data = JSON.parse(responseBodyText);
            if (!data.choices || data.choices.length === 0 || !data.choices[0].message) {
                 throw new Error(`Invalid response structure from ${providerName}`);
            }
            responseText = data.choices[0].message.content;
        } catch (e: any) {
            console.error(`Error parsing JSON from ${providerName}:`, responseBodyText);
            throw new Error(`Failed to parse JSON from ${providerName}: ${e.message}`);
        }

      } else {
        throw new Error(`Unsupported provider: ${provider}`);
      }

      return responseText;

    } catch (error) {
      console.error(`Attempt ${i + 1} failed for ${provider}:`, error);
      if (i === retries) {
        if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
             throw new Error(`网络请求失败。这可能是由于 CORS 策略、网络连接中断或代理服务器配置错误。请检查浏览器开发者工具中的网络(Network)和控制台(Console)选项卡以获取详细信息。`);
        }
        throw error; // Re-throw the last error
      }
      await new Promise(res => setTimeout(res, 1000));
    }
  }
  throw new Error('All retry attempts failed.');
};

// New function for streaming chat responses
const callGenerativeAiStream = async (
    provider: ModelProvider,
    systemInstruction: string,
    userPrompt: string,
    apiKeys: { [key in ModelProvider]?: string },
    history: ChatMessage[],
    onChunk: (textChunk: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void,
    thinkingBudget?: number,
) => {
    try {
        if (provider !== 'gemini' && !apiKeys[provider]) {
            throw new Error(`API key for ${provider} is missing.`);
        }

        if (provider === 'gemini') {
            const geminiApiKey = process.env.API_KEY;
            if (!geminiApiKey || geminiApiKey.trim() === '' || geminiApiKey === 'undefined') {
                throw new Error("Gemini API key 未配置。请在您的 .env 文件中设置 VITE_GEMINI_API_KEY 变量。如果部署在 GitHub Pages, 请确保已在仓库的 Secrets 中正确配置 VITE_GEMINI_API_KEY。");
            }
            const ai = new GoogleGenAI({ apiKey: geminiApiKey });
            const config: any = { systemInstruction };
            
            if (thinkingBudget !== undefined) {
              config.thinkingConfig = { thinkingBudget };
            }
            
            const fullContents = [...history.filter(h => h.parts && h.parts.length > 0), { role: 'user', parts: [{ text: userPrompt }] }];
            const contentsForApi = fullContents.map(({ role, parts }) => ({ role, parts }));

            const responseStream = await ai.models.generateContentStream({
                model: 'gemini-2.5-flash',
                contents: contentsForApi,
                config: config
            });

            for await (const chunk of responseStream) {
                onChunk(chunk.text ?? '');
            }
        } else if (provider === 'openai' || provider === 'deepseek' || provider === 'ali') {
            const { endpoint, model } = getModelConfig(provider);
            if (!endpoint || !model) {
                throw new Error(`Configuration for provider ${provider} is missing. Please check your .env file.`);
            }
            
            const transformedHistory = history.filter(h => h.parts && h.parts.length > 0).map(msg => ({
              role: msg.role === 'model' ? 'assistant' : 'user',
              content: msg.parts[0].text
            }));

            const body = {
                model,
                messages: [{ role: 'system', content: systemInstruction }, ...transformedHistory, { role: 'user', content: userPrompt }],
                stream: true,
                max_tokens: 4096
            };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKeys[provider]}` },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                 const errorText = await response.text();
                 let errorMessage = errorText;
                 if (response.status >= 500 && response.status < 600) {
                     errorMessage = `代理服务器连接失败 (状态码: ${response.status})。这通常是由于网络问题、VPN配置错误或目标API服务暂时不可用导致的。请检查您的网络连接并重试。`;
                 } else if (errorText) {
                     try {
                         const errorJson = JSON.parse(errorText);
                         errorMessage = errorJson.error?.message || errorMessage;
                     } catch (e) { /* Ignore if parsing fails */ }
                 }
                 throw new Error(`API 错误: ${response.status} ${response.statusText} - ${errorMessage}`);
            }

            if (!response.body) {
                throw new Error("Response body is null");
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep the last partial line in the buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        if (data === '[DONE]') {
                            // Stream finished
                        } else {
                            try {
                                const parsed = JSON.parse(data);
                                const textChunk = parsed.choices?.[0]?.delta?.content || '';
                                if (textChunk) {
                                    onChunk(textChunk);
                                }
                            } catch (e) {
                                console.error('Error parsing stream data:', e);
                            }
                        }
                    }
                }
            }
        }
        onComplete();
    } catch (error: any) {
        if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
            onError(new Error(`网络请求失败。这可能是由于 CORS 策略、网络连接中断或代理服务器配置错误。请检查浏览器开发者工具中的网络(Network)和控制台(Console)选项卡以获取详细信息。`));
        } else {
            onError(error);
        }
    }
};

const ThoughtsInputModal = ({
  isOpen,
  onClose,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (thoughts: string) => void;
}) => {
  const [thoughts, setThoughts] = useState('');

  if (!isOpen) return null;

  const handleSubmit = () => {
    onSubmit(thoughts);
    setThoughts(''); // Reset for next time
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>输入我的想法</h2>
        <p>在整理笔记前，您可以输入任何相关的想法、问题或待办事项。AI 会将这些内容与您的笔记一并智能整理。</p>
        <textarea
          className="modal-textarea"
          rows={5}
          value={thoughts}
          onChange={(e) => setThoughts(e.target.value)}
          placeholder="例如：这个概念需要进一步查证，下周三前完成..."
          autoFocus
        />
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            取消
          </button>
          <button className="btn btn-primary" onClick={handleSubmit}>
            开始整理
          </button>
        </div>
      </div>
    </div>
  );
};


const ApiKeyModal = ({
  isOpen,
  onClose,
  onSave,
  apiKeys,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSave: (keys: { [key in ModelProvider]?: string }) => void;
  apiKeys: { [key in ModelProvider]?: string };
}) => {
  const [keys, setKeys] = useState(apiKeys);

  useEffect(() => {
    setKeys(apiKeys);
  }, [apiKeys, isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(keys);
  };
  
  const providers: {id: ModelProvider, name: string}[] = [
      {id: 'openai', name: 'OpenAI'},
      {id: 'deepseek', name: 'DeepSeek'},
      {id: 'ali', name: 'Ali'}
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>设置 API Keys</h2>
        <p>请输入您在各个平台申请的 API Key，以便使用对应的模型。输入的内容将仅保存在您的浏览器中。</p>
        {providers.map(p => (
           <div key={p.id}>
              <h4 style={{marginBottom: '8px'}}>{p.name}</h4>
              <input
                type="password"
                className="modal-textarea" // Re-using style
                value={keys[p.id] || ''}
                onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })}
                placeholder={`请输入 ${p.name} API Key`}
              />
            </div>
        ))}
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            取消
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            保存
          </button>
        </div>
      </div>
    </div>
  );
};

const HomeInputView = ({
  inputText,
  setInputText,
  onOrganize,
  onAudit,
  selectedModel,
  setSelectedModel,
  onOpenApiModal,
  apiKeys,
  isProcessing,
  knowledgeBases,
  isKbLoading,
  kbError,
  selectedKnowledgeBase,
  setSelectedKnowledgeBase,
  onKnowledgeChat,
  onWriting,
}: {
  inputText: string;
  setInputText: React.Dispatch<React.SetStateAction<string>>;
  onOrganize: () => void;
  onAudit: () => void;
  selectedModel: ModelProvider;
  setSelectedModel: (model: ModelProvider) => void;
  onOpenApiModal: () => void;
  apiKeys: { [key in ModelProvider]?: string };
  isProcessing: boolean;
  knowledgeBases: { id: string; name: string }[];
  isKbLoading: boolean;
  kbError: string | null;
  selectedKnowledgeBase: string | null;
  setSelectedKnowledgeBase: (id: string) => void;
  onKnowledgeChat: () => void;
  onWriting: () => void;
}) => {
    const lastPastedText = useRef('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const handleFocus = async () => {
            if (document.hasFocus()) {
                try {
                    const text = await navigator.clipboard.readText();
                    if (text && text !== lastPastedText.current && text !== inputText) {
                        setInputText(prev => prev ? `${prev}\n\n${text}` : text);
                        lastPastedText.current = text;
                    }
                } catch (err) {
                    console.log('Clipboard permission denied, or clipboard is empty.');
                }
            }
        };

        window.addEventListener('focus', handleFocus);

        return () => {
            window.removeEventListener('focus', handleFocus);
        };
    }, [inputText, setInputText]);

  const processFile = async (file: File) => {
    if (!file) return;
    const reader = new FileReader();

    reader.onload = async (event) => {
        const fileContent = event.target?.result;
        let text = '';
        if (file.name.endsWith('.docx')) {
            try {
                const result = await mammoth.extractRawText({ arrayBuffer: fileContent as ArrayBuffer });
                text = result.value;
            } catch (err) {
                console.error("Error reading docx file", err);
                alert("无法解析 DOCX 文件。");
                return;
            }
        } else {
            text = fileContent as string;
        }
        setInputText(prev => prev ? `${prev}\n\n--- ${file.name} ---\n${text}` : text);
    };
    
    if (file.name.endsWith('.docx')) {
        reader.readAsArrayBuffer(file);
    } else if (file.name.endsWith('.txt') || file.name.endsWith('.md')) {
        reader.readAsText(file);
    } else {
        alert("不支持的文件类型。请上传 .txt, .md 或 .docx 文件。");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
        processFile(e.target.files[0]);
        e.target.value = '';
    }
  };

  const handleUploadClick = () => {
      fileInputRef.current?.click();
  };

  const handleDragOver = (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
  };

  const handleDragLeave = (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
  };

  const handleDrop = async (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');

    if (e.dataTransfer.files?.[0]) {
      await processFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  };

  const modelProviders: ModelProvider[] = ['gemini', 'openai', 'deepseek', 'ali'];

  return (
    <>
        <div className="home-grid-layout">
            <div className="home-panel">
                <h2>工作区</h2>
                <textarea
                    className="text-area"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    placeholder="在此处输入或拖放 .txt, .md, .docx 文件...&#10;从别处复制后，返回此页面可自动粘贴"
                    disabled={isProcessing}
                    style={{flexGrow: 1}}
                />
                 <input
                    type="file"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    accept=".txt,.md,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    onChange={handleFileChange}
                />
                <div className="utility-btn-group">
                    <button className="btn btn-secondary" onClick={() => setInputText('')} disabled={!inputText || isProcessing}>
                        清空内容
                    </button>
                    <button className="btn btn-secondary" onClick={handleUploadClick} disabled={isProcessing}>
                        上传文件
                    </button>
                </div>
            </div>
            <div className="home-panel">
                <h2>全局配置</h2>
                <div className="config-group">
                    <h4>选择模型</h4>
                    <div className="model-selector-group">
                        {modelProviders.map(model => (
                            <button
                                key={model}
                                className={`model-btn ${selectedModel === model ? 'active' : ''} ${model !== 'gemini' && !apiKeys[model] ? 'disabled' : ''}`}
                                onClick={() => setSelectedModel(model)}
                                disabled={(model !== 'gemini' && !apiKeys[model]) || isProcessing}
                                title={model !== 'gemini' && !apiKeys[model] ? `请先设置 ${model} API Key` : ''}
                            >
                                {model}
                            </button>
                        ))}
                    </div>
                </div>
                <div className="config-group">
                    <h4>选择知识库</h4>
                    {isKbLoading && <div className="spinner-container" style={{padding: '10px 0'}}><p>正在加载知识库...</p></div>}
                    {kbError && <div className="error-message" style={{textAlign: 'left'}}>{kbError}</div>}
                    {!isKbLoading && !kbError && (
                        knowledgeBases.length > 0 ? (
                            <div className="kb-selector-group">
                                {knowledgeBases.map(kb => (
                                    <button
                                        key={kb.id}
                                        className={`kb-selector-btn ${selectedKnowledgeBase === kb.id ? 'active' : ''}`}
                                        onClick={() => setSelectedKnowledgeBase(kb.id)}
                                        disabled={isProcessing}
                                    >
                                        {kb.name}
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <p className="instruction-text">未找到可用的知识库。请检查后端服务和 Milvus 连接。</p>
                        )
                    )}
                </div>
                <div className="config-group">
                    <h4>API Keys</h4>
                    <button className="btn btn-secondary" onClick={onOpenApiModal} style={{ width: '100%' }}>
                        设置 API Keys
                    </button>
                </div>
            </div>
        </div>
        <div className="home-actions-bar">
            <button className="action-btn" onClick={onOrganize} disabled={!inputText || isProcessing}>
                1. 整理笔记
            </button>
            <button className="action-btn" onClick={onAudit} disabled={!inputText || isProcessing}>
                2. 审阅文本
            </button>
            <button className="action-btn" onClick={onKnowledgeChat} disabled={!inputText || isProcessing || !selectedKnowledgeBase}>
                3. 知识库对话
            </button>
            <button className="action-btn" onClick={onWriting} disabled={isProcessing}>
                4. 沉浸式写作
            </button>
        </div>
    </>
  );
};


const NoteAnalysisView = ({
  analysisResult,
  isLoading: isInitialLoading,
  error,
  provider,
  originalText,
  apiKeys,
  selectedKnowledgeBaseId,
  knowledgeBases
}: {
  analysisResult: NoteAnalysis | null;
  isLoading: boolean;
  error: string | null;
  provider: ModelProvider;
  originalText: string;
  apiKeys: { [key in ModelProvider]?: string };
  selectedKnowledgeBaseId: string | null;
  knowledgeBases: { id: string; name: string }[];
}) => {
  const [consolidatedText, setConsolidatedText] = useState('');
  
  // State for Chat
  const [chatHistory, setChatHistory] = useState<NoteChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatHistoryRef = useRef<HTMLDivElement>(null);

  // State for Roaming Notes
  const [isRoaming, setIsRoaming] = useState(false);
  const [roamingResult, setRoamingResult] = useState<RoamingResultItem[] | null>(null);
  const [roamingError, setRoamingError] = useState<string | null>(null);


  useEffect(() => {
    if (analysisResult) {
      const fullText = `【整理后】\n${analysisResult.organizedText}\n\n---\n\n【我的想法】\n${analysisResult.userThoughts}\n\n---\n\n【原文】\n${originalText}`;
      setConsolidatedText(fullText);
      // Initialize chat with a welcome message
      setChatHistory([{ role: 'model', text: '您好！您可以针对这篇笔记进行提问、要求修改，或者探讨更多想法。' }]);
    }
  }, [analysisResult, originalText]);

  useEffect(() => {
    if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleExportTXT = () => {
    if (!analysisResult) return;

    // Part 1: Main Content
    let content = `【笔记工作台】\n\n【整理后】\n${analysisResult.organizedText}\n\n---\n\n【我的想法】\n${analysisResult.userThoughts}`;
    
    // Part 2: Roaming Result
    if (roamingResult && roamingResult.length > 0) {
      content += `\n\n---\n\n【笔记漫游】`;
      roamingResult.forEach((result: RoamingResultItem, index: number) => {
        content += `\n\n--- 漫游结果 ${index + 1} ---\n`;
        content += `来源: ${result.source}\n\n`;
        content += `关联原文:\n${result.relevantText}\n\n`;
        content += `联想结论:\n${result.conclusion}`;
      });
    }
    
    // Part 3: Original Text
    content += `\n\n---\n\n【原文】\n${originalText}`;

    // Part 4: Chat History
    const chatContent = chatHistory.map(msg => {
        // Skip the initial welcome message from the model
        if (msg.role === 'model' && msg.text.startsWith('您好！您可以针对这篇笔记进行提问')) {
            return '';
        }
        const role = msg.role === 'user' ? 'User' : 'Model';
        return `[${role}]\n${msg.text}`;
    }).filter(Boolean).join('\n\n');

    if (chatContent) {
        content += `\n\n---\n\n【多轮问答】\n${chatContent}`;
    }

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `笔记整理与讨论 - ${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleStartRoaming = async () => {
    if (!selectedKnowledgeBaseId || !analysisResult) {
        if (!selectedKnowledgeBaseId) {
            alert("请返回首页选择一个知识库以开始笔记漫游。");
        }
        return;
    }

    setIsRoaming(true);
    setRoamingError(null);
    setRoamingResult(null);

    try {
        // Step 1: Call local backend to get relevant context
        const backendResponse = await fetch('/api/find-related', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: analysisResult.organizedText,
                collection_name: selectedKnowledgeBaseId,
                top_k: 3
            })
        });

        if (!backendResponse.ok) {
            const errorText = await backendResponse.text().catch(() => backendResponse.statusText);
            let errorJson;
            if (errorText) {
                try {
                    errorJson = JSON.parse(errorText);
                } catch (e) { /* Not JSON */ }
            }
            throw new Error(`知识库查询失败: ${errorJson?.error || errorText}`);
        }

        const responseText = await backendResponse.text();
        if (!responseText) {
            throw new Error("知识库查询返回为空。");
        }
        
        let backendData;
        try {
            backendData = JSON.parse(responseText);
        } catch (e: any) {
            console.error('Error parsing backend JSON:', responseText);
            throw new Error(`Backend returned invalid JSON: ${e.message}`);
        }
        
        if (backendData.error) {
            throw new Error(`知识库返回错误: ${backendData.error}`);
        }
        
        const sources: Source[] = backendData.related_documents || [];

        if (sources.length === 0) {
            setRoamingError("在知识库中未找到足够相关的内容来进行漫游联想。");
            setIsRoaming(false);
            return;
        }

        // Step 2: Call Generative AI for each source to create a conclusion
        const systemInstruction = `You are an AI assistant skilled at synthesizing information. Based on a user's note and a relevant passage from their knowledge base, create an "Associative Conclusion" connecting the two ideas. Your entire response must be a JSON object with one key: "conclusion" (your generated associative summary).`;
        
        const roamingPromises = sources.map(async (source: Source) => {
            const userPrompt = `[Relevant Passage from Knowledge Base]:\n${source.content_chunk}\n\n[User's Original Note]:\n${analysisResult.organizedText}`;
            const genAiResponseText = await callGenerativeAi(provider, systemInstruction, userPrompt, true, apiKeys, 'roaming');
            const result = JSON.parse(genAiResponseText.replace(/```json\n?|\n?```/g, ''));

            if (!result.conclusion) {
                throw new Error("AI model did not return a valid conclusion for one of the documents.");
            }
            
            return {
                source: source.source_file,
                relevantText: source.content_chunk,
                conclusion: result.conclusion,
            };
        });

        const newRoamingResults = await Promise.all(roamingPromises);
        setRoamingResult(newRoamingResults);

    } catch (err: any) {
        setRoamingError(`笔记漫游失败: ${err.message}`);
    } finally {
        setIsRoaming(false);
    }
  };
  
  const handleSendChatMessage = async (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!chatInput.trim() || isChatLoading || !analysisResult) return;

      const newUserMessage: NoteChatMessage = { role: 'user', text: chatInput };
      const currentChatHistory = [...chatHistory, newUserMessage];
      setChatHistory(currentChatHistory);
      setChatInput('');
      setIsChatLoading(true);

      const systemInstruction = `You are a helpful assistant. The user has just organized a note and wants to discuss it. The note's organized content is provided below. Your role is to answer questions, help refine the text, or brainstorm ideas based on this note. Be helpful and conversational.\n\n--- NOTE START ---\n${analysisResult.organizedText}\n--- NOTE END ---`;
      
      const chatHistoryForApi = currentChatHistory.slice(0, -1).map(msg => ({ // Exclude the user message we just added
          role: msg.role as 'user' | 'model',
          parts: [{ text: msg.text }]
      }));

      const modelResponse: NoteChatMessage = { role: 'model', text: '' };
      setChatHistory(prev => [...prev, modelResponse]);

      try {
          await callGenerativeAiStream(
              provider,
              systemInstruction,
              chatInput,
              apiKeys,
              chatHistoryForApi,
              (chunk) => {
                  setChatHistory(prev => {
                      const newHistory = [...prev];
                      if(newHistory.length > 0) {
                        newHistory[newHistory.length - 1].text += chunk;
                      }
                      return newHistory;
                  });
              },
              () => {
                  setIsChatLoading(false);
              },
              (error) => {
                  setChatHistory(prev => {
                      const newHistory = [...prev];
                      if(newHistory.length > 0) {
                        newHistory[newHistory.length - 1].text = `抱歉，出错了: ${error.message}`;
                        newHistory[newHistory.length - 1].isError = true;
                      }
                      return newHistory;
                  });
                  setIsChatLoading(false);
              }
          );
      } catch (error: any) {
           setChatHistory(prev => {
              const newHistory = [...prev];
              if(newHistory.length > 0) {
                newHistory[newHistory.length - 1].text = `抱歉，出错了: ${error.message}`;
                newHistory[newHistory.length - 1].isError = true;
              }
              return newHistory;
          });
          setIsChatLoading(false);
      }
  };

  if (isInitialLoading) {
      return (
          <div className="spinner-container">
              <div className="spinner large"></div>
              <p style={{ marginTop: '16px', color: '#a0a0a0' }}>正在整理，请稍候...</p>
          </div>
      );
  }
  if (error) {
      return <div className="error-message" style={{ textAlign: 'left', whiteSpace: 'pre-wrap' }}>{error}</div>;
  }
  if (!analysisResult) {
      return <div className="large-placeholder">分析结果将显示在此处。</div>;
  }

  return (
      <div className="note-analysis-layout">
        <div className="note-content-panel">
            <h2 style={{textTransform: 'capitalize'}}>笔记工作台 (由 {provider} 模型生成)</h2>
            <div className="note-content-scrollable-area">
                <textarea
                    readOnly
                    className="text-area consolidated-note-display"
                    value={consolidatedText}
                ></textarea>
                <div className="content-section" style={{padding: '16px', backgroundColor: 'var(--background-color)'}}>
                    <h3>笔记漫游</h3>
                    {!roamingResult && !isRoaming && !roamingError && <p className="instruction-text">如需基于笔记内容进行关联联想，请在首页选择知识库后，点击下方“开始笔记漫游”按钮。</p>}
                    {isRoaming && <div className="spinner-container" style={{padding: '20px 0'}}><div className="spinner"></div></div>}
                    {roamingError && <div className="error-message">{roamingError}</div>}
                    {roamingResult && (
                        <div className="roaming-results-container">
                            {roamingResult.map((result: RoamingResultItem, index: number) => (
                                <div key={index} className="roaming-result">
                                    <p><strong>来源 ({index + 1}):</strong> {result.source}</p>
                                    <p><strong>关联原文:</strong> {result.relevantText}</p>
                                    <p><strong>联想结论:</strong> {result.conclusion}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
            <div className="card-bottom-actions">
                <div className="button-group">
                    <button className="btn btn-secondary" onClick={handleStartRoaming} disabled={isRoaming || !selectedKnowledgeBaseId}>
                        {isRoaming ? '漫游中...' : `开始笔记漫游 (使用 ${provider})`}
                    </button>
                </div>
                <div className="button-group" style={{marginLeft: 'auto'}}>
                    <button className="btn btn-secondary" onClick={handleExportTXT}>导出 TXT</button>
                </div>
            </div>
        </div>
        <div className="note-chat-panel">
            <h2>多轮问答</h2>
            <div className="kb-chat-history" ref={chatHistoryRef}>
                {chatHistory.map((msg, index) => (
                    <div key={index} className={`kb-message ${msg.role} ${msg.isError ? 'error' : ''}`}>
                        <p>{msg.text}</p>
                    </div>
                ))}
                {isChatLoading && chatHistory[chatHistory.length - 1]?.role === 'model' && !chatHistory[chatHistory.length - 1]?.text && (
                    <div className="spinner-container" style={{padding: '10px 0'}}><div className="spinner"></div></div>
                )}
            </div>
            <form className="chat-input-form" onSubmit={handleSendChatMessage}>
                <textarea
                    className="chat-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendChatMessage(); }}}
                    placeholder="针对笔记提问..."
                    rows={2}
                    disabled={isChatLoading}
                />
                <button type="submit" className="btn btn-primary send-btn" disabled={isChatLoading || !chatInput.trim()}>发送</button>
            </form>
        </div>
      </div>
  );
};

const parseJsonResponse = <T,>(responseText: string): { data: T | null, error?: string, rawResponse?: string } => {
    let parsedData: T | null = null;
    let jsonString = responseText.trim();

    const tryParse = (str: string): T | null => {
        try {
            const fixedStr = str.replace(/,\s*([}\]])/g, '$1');
            const result = JSON.parse(fixedStr);
            return result as T;
        } catch {
            try {
                const result = JSON.parse(str);
                return result as T;
            } catch {
                return null;
            }
        }
    };

    parsedData = tryParse(jsonString);

    if (!parsedData) {
        const markdownMatch = jsonString.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (markdownMatch && markdownMatch[1]) {
            parsedData = tryParse(markdownMatch[1].trim());
        }
    }

    if (!parsedData) {
        const firstBrace = jsonString.indexOf('{');
        const lastBrace = jsonString.lastIndexOf('}');
        const firstBracket = jsonString.indexOf('[');
        const lastBracket = jsonString.lastIndexOf(']');

        let startIndex = -1, endIndex = -1;
        
        if (firstBracket !== -1 && lastBracket > firstBracket) {
            startIndex = firstBracket;
            endIndex = lastBracket;
        } else if (firstBrace !== -1 && lastBrace > firstBrace) {
            startIndex = firstBrace;
            endIndex = lastBrace;
        }
        
        if (startIndex !== -1) {
            parsedData = tryParse(jsonString.substring(startIndex, endIndex + 1));
        }
    }

    if (parsedData === null) {
        const lowercasedResponse = responseText.toLowerCase();
        if (Array.isArray([] as T) && (lowercasedResponse.includes('no issues found') || lowercasedResponse.includes('没有发现') || lowercasedResponse.includes('未发现'))) {
            return { data: [] as T };
        }
        return { 
            data: null, 
            error: "未能将模型响应解析为有效的JSON。", 
            rawResponse: responseText 
        };
    }
    return { data: parsedData };
};


const parseAuditResponse = (responseText: string): { issues: AuditIssue[], error?: string, rawResponse?: string } => {
    const { data, error, rawResponse } = parseJsonResponse<AuditIssue[]>(responseText);

    if (!data) {
        return { issues: [], error, rawResponse };
    }

    const validIssues = data.filter(issue => 
        issue && 
        typeof issue.problematicText === 'string' &&
        typeof issue.suggestion === 'string' &&
        typeof issue.checklistItem === 'string' &&
        typeof issue.explanation === 'string' &&
        issue.problematicText.trim()
    );
    return { issues: validIssues };
};

const AuditView = ({
    initialText,
    apiKeys,
    selectedModel
} : {
    initialText: string;
    apiKeys: { [key in ModelProvider]?: string };
    selectedModel: ModelProvider;
}) => {
    const [text] = useState(initialText);
    const [auditResults, setAuditResults] = useState<AuditResults>({});
    const [isLoading, setIsLoading] = useState(false);
    const [checklist, setChecklist] = useState<string[]>([
        '全文错别字',
        '全文中文语法问题',
        '文中逻辑不合理的地方',
        '学术名词是否前后一致'
    ]);
    const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null);
    const textDisplayRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (selectedIssueId && textDisplayRef.current) {
            const element = textDisplayRef.current.querySelector(`[data-issue-id="${selectedIssueId}"]`);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }, [selectedIssueId]);

    const handleChecklistItemChange = (index: number, value: string) => {
        const newChecklist = [...checklist];
        newChecklist[index] = value;
        setChecklist(newChecklist);
    };

    const addChecklistItem = () => setChecklist([...checklist, '']);
    const removeChecklistItem = (index: number) => setChecklist(checklist.filter((_, i) => i !== index));

    const handleAudit = async () => {
        setIsLoading(true);
        setAuditResults({});
        setSelectedIssueId(null);
        const model = selectedModel;

        const systemInstruction = `You are a professional editor. Analyze the provided text based ONLY on the rules in the following checklist. For each issue you find, return a JSON object with "problematicText" (the exact, verbatim text segment from the original), "suggestion" (your proposed improvement), "checklistItem" (the specific rule from the checklist that was violated), and "explanation" (a brief explanation of why it's a problem). Your entire response MUST be a single JSON array of these objects, or an empty array [] if no issues are found.

[Checklist]:
- ${checklist.filter(item => item.trim()).join('\n- ')}
`;
        const userPrompt = `[Text to Audit]:\n\n${text}`;
        
        try {
            const responseText = await callGenerativeAi(model, systemInstruction, userPrompt, true, apiKeys, 'audit');
            const { issues, error, rawResponse } = parseAuditResponse(responseText);
            setAuditResults({ [model]: { issues, error, rawResponse } });
            
        } catch (err: any) {
            console.error(`Error auditing with ${model}:`, err);
            setAuditResults({ [model]: { issues: [], error: err.message } });
        } finally {
            setIsLoading(false);
        }
    };

    const handleAuditAll = async () => {
        setIsLoading(true);
        setAuditResults({});
        setSelectedIssueId(null);

        const allModels: ModelProvider[] = ['gemini', 'openai', 'deepseek', 'ali'];
        const enabledModels = allModels.filter(m => m === 'gemini' || (apiKeys[m] && apiKeys[m]?.trim() !== ''));

        if (enabledModels.length === 0) {
            setIsLoading(false);
            return;
        }

        const systemInstruction = `You are a professional editor. Analyze the provided text based ONLY on the rules in the following checklist. For each issue you find, return a JSON object with "problematicText" (the exact, verbatim text segment from the original), "suggestion" (your proposed improvement), "checklistItem" (the specific rule from the checklist that was violated), and "explanation" (a brief explanation of why it's a problem). Your entire response MUST be a single JSON array of these objects, or an empty array [] if no issues are found.

[Checklist]:
- ${checklist.filter(item => item.trim()).join('\n- ')}
`;
        const userPrompt = `[Text to Audit]:\n\n${text}`;

        const auditPromises = enabledModels.map(model => 
            callGenerativeAi(model, systemInstruction, userPrompt, true, apiKeys, 'audit')
        );
        
        const results = await Promise.allSettled(auditPromises);
        
        const newAuditResults: AuditResults = {};
        results.forEach((result, index) => {
            const model = enabledModels[index];
            if (result.status === 'fulfilled') {
                const { issues, error, rawResponse } = parseAuditResponse(result.value);
                newAuditResults[model] = { issues, error, rawResponse };
            } else {
                newAuditResults[model] = { issues: [], error: (result.reason as Error).message };
            }
        });

        setAuditResults(newAuditResults);
        setIsLoading(false);
    };

    // FIX: Explicitly cast the result of Object.entries to fix type inference
    // issues where 'result' was being inferred as 'unknown'.
    const allIssuesWithIds = (Object.entries(auditResults) as [string, AuditResult | undefined][]).flatMap(([model, result]) => {
        return result?.issues?.map((issue: AuditIssue, index: number) => ({
            ...issue,
            model: model as ModelProvider,
            id: `${model}-${index}`
        })) ?? [];
    });

    const renderOriginalTextWithHighlight = () => {
        if (!text) return <div className="large-placeholder">审阅结果将显示在此处。</div>;
        const selectedIssue = selectedIssueId ? allIssuesWithIds.find(i => i.id === selectedIssueId) : null;
        if (!selectedIssue) {
            return <div className="audit-text-display">{text}</div>;
        }
        const term = selectedIssue.problematicText;
        if (!term || term.trim() === '') {
             return <div className="audit-text-display">{text}</div>;
        }
        try {
            const regex = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'g');
            const parts = text.split(regex);
            let firstMatch = true;
            return (
                <div className="audit-text-display">
                    {parts.map((part, index) => {
                        if (part === term) {
                            const idToAssign = firstMatch ? selectedIssue.id : undefined;
                            firstMatch = false;
                            return (
                                <span key={index} className="selected-highlight" data-issue-id={idToAssign}>
                                    {part}
                                </span>
                            );
                        }
                        return <React.Fragment key={index}>{part}</React.Fragment>;
                    })}
                </div>
            );
        } catch (e) {
            console.error("Regex error in highlighting:", e);
            return <div className="audit-text-display">{text}</div>;
        }
    };

    // FIX: Explicitly cast the result of Object.values to fix type inference
    // issues where 'res' was being inferred as 'unknown'.
    const hasAnyIssues = (Object.values(auditResults) as (AuditResult | undefined)[]).some(res => (res?.issues?.length ?? 0) > 0);
    const hasAnyErrors = (Object.values(auditResults) as (AuditResult | undefined)[]).some(res => !!(res?.error));

    return (
        <div className="audit-view-container">
            <div className="audit-config-panel">
                <h2 style={{textTransform: 'capitalize'}}>审阅清单</h2>
                <div className="checklist-editor">
                    {checklist.map((item, index) => (
                        <div key={index} className="checklist-item">
                            <input
                                type="text"
                                value={item}
                                onChange={(e) => handleChecklistItemChange(index, e.target.value)}
                                placeholder={`规则 #${index + 1}`}
                                disabled={isLoading}
                            />
                            <button onClick={() => removeChecklistItem(index)} disabled={isLoading}>-</button>
                        </div>
                    ))}
                    <button className="btn btn-secondary" onClick={addChecklistItem} disabled={isLoading}>+ 添加规则</button>
                </div>
                <div className="audit-button-group">
                     <button className="btn btn-primary audit-start-btn" onClick={handleAudit} disabled={isLoading || !text}>
                        {isLoading ? <span className="spinner"></span> : null}
                        {isLoading ? '审阅中...' : `审阅 (${selectedModel})`}
                    </button>
                    <button className="btn btn-primary audit-start-btn" onClick={handleAuditAll} disabled={isLoading || !text}>
                        {isLoading ? <span className="spinner"></span> : null}
                        {isLoading ? '审阅中...' : '全部模型审阅'}
                    </button>
                </div>
                <div className="audit-status-area">
                    {/* FIX: Explicitly cast the result of Object.entries to fix type inference
                    // issues where 'result' was being inferred as 'unknown'. */}
                    { (Object.entries(auditResults) as [string, AuditResult | undefined][]).map(([model, result]: [string, AuditResult | undefined]) => {
                        if (!result) return null;
                        return (
                        <div key={model} className="audit-status-item">
                            <span className={`model-indicator model-${model}`}>{model}</span>
                            {result.error 
                                ? <span className="status-error">失败: {result.error}</span>
                                : <span className="status-success">完成 ({result.issues.length}个问题)</span>
                            }
                        </div>
                    )})}
                </div>
            </div>

            <div className="audit-results-panel">
                <div className="content-section audit-original-text-section">
                    <h2>原始文本</h2>
                    <div className="original-text-container" ref={textDisplayRef}>
                       {isLoading && !Object.keys(auditResults).length ? <div className="spinner-container"><div className="spinner large"></div><p>正在调用模型，请稍候...</p></div> : renderOriginalTextWithHighlight()}
                    </div>
                </div>
                <div className="content-section audit-issues-section">
                    <h2>审核问题</h2>
                    <div className="issues-list-container">
                        {!isLoading && Object.keys(auditResults).length > 0 && !hasAnyIssues && !hasAnyErrors && <div className="large-placeholder">未发现任何问题。</div>}
                        {/* FIX: Explicitly cast the result of Object.entries to fix type inference
                        // issues where 'result' was being inferred as 'unknown'. */}
                        { (Object.entries(auditResults) as [string, AuditResult | undefined][]).map(([model, result]: [string, AuditResult | undefined]) => {
                            if (!result) return null;

                            if (result.error && result.rawResponse) {
                                return (
                                    <details key={`${model}-error`} open className="issue-group">
                                        <summary className={`issue-group-summary model-border-${model}`}>
                                            <span className={`model-indicator model-${model}`}>{model}</span> (解析失败)
                                        </summary>
                                        <div className="issue-group-content">
                                            <div className="issue-card raw-response-card">
                                                <div className="issue-card-header">原始模型响应 (Raw Model Response)</div>
                                                <div className="issue-card-body">
                                                    <pre className="raw-response-text">{result.rawResponse}</pre>
                                                </div>
                                            </div>
                                        </div>
                                    </details>
                                );
                            }

                            if (result.issues.length === 0) return null;
                            
                            return (
                                <details key={model} open className="issue-group">
                                    <summary className={`issue-group-summary model-border-${model}`}>
                                        <span className={`model-indicator model-${model}`}>{model}</span> ({result.issues.length}个问题)
                                    </summary>
                                    <div className="issue-group-content">
                                    {result.issues.map((issue: AuditIssue, index: number) => {
                                        const issueId = `${model}-${index}`;
                                        return (
                                            <div
                                                key={issueId}
                                                className={`issue-card ${selectedIssueId === issueId ? 'selected' : ''}`}
                                                onClick={() => setSelectedIssueId(issueId)}
                                                tabIndex={0}
                                                onKeyDown={(e) => { if(e.key === 'Enter' || e.key === ' ') setSelectedIssueId(issueId)}}
                                            >
                                                <div className="issue-card-header">{issue.checklistItem}</div>
                                                <div className="issue-card-body">
                                                    <p><strong>原文:</strong> {issue.problematicText}</p>
                                                    <p><strong>建议:</strong> {issue.suggestion}</p>
                                                    <p><strong>说明:</strong> {issue.explanation}</p>
                                                </div>
                                            </div>
                                        );
                                    })}
                                    </div>
                                </details>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};

// By moving this pure function outside the component, we prevent it from being
// recreated on every render, which is a minor performance optimization.
const parseMessageText = (text: string) => {
    if (!text) return '';
    const textWithCitations = text.replace(/\[Source: (.*?)\]/g, (match, filename) => {
        return `<a href="#" class="source-citation" data-filename="${filename.trim()}">${match}</a>`;
    });
    return marked.parse(textWithCitations, { gfm: true, breaks: true }) as string;
};

const KnowledgeChatView = ({
  knowledgeBaseId,
  knowledgeBaseName,
  initialQuestion,
  provider,
  apiKeys,
}: {
  knowledgeBaseId: string;
  knowledgeBaseName: string;
  initialQuestion?: string;
  provider: ModelProvider;
  apiKeys: { [key in ModelProvider]?: string };
}) => {
  const [chatHistory, setChatHistory] = useState<NoteChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const isInitialQuestionSent = useRef(false);

  useEffect(() => {
    setChatHistory([{ role: 'model', text: `您好！已连接到“${knowledgeBaseName}”知识库。每次提问我都会优先从知识库中寻找答案。`, isComplete: true }]);
  }, [knowledgeBaseName]);
  
  useEffect(() => {
    if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);
  
  const handleExportTXT = () => {
    const content = chatHistory.map(msg => {
        if (msg.role === 'model' && msg.text.startsWith('您好！已连接到')) {
            return '';
        }
        let entry = '';
        if (msg.role === 'user') {
            entry += `[User]\n${msg.text}\n\n`;
        } else { // model
            entry += `[Model]\n${msg.text}\n`;
            if (msg.sources && msg.sources.length > 0) {
                entry += `\n--- 参考源头信息 ---\n`;
                msg.sources.forEach(source => {
                    entry += `  - 文件: ${source.source_file}\n`;
                    entry += `    Relevance: ${source.score.toFixed(2)}\n`;
                    entry += `    内容片段: "${source.content_chunk}"\n\n`;
                });
            } else {
                entry += '\n';
            }
        }
        return entry;
    }).filter(Boolean).join('---\n\n');

    if (!content.trim()) {
        alert("没有可导出的对话内容。");
        return;
    }

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `知识库对话 - ${knowledgeBaseName} - ${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  const handleSendChatMessage = useCallback(async (e?: React.FormEvent, messageOverride?: string) => {
    e?.preventDefault();
    const messageToSend = messageOverride || chatInput;
    if (!messageToSend.trim() || isChatLoading) return;

    const newUserMessage: NoteChatMessage = { role: 'user', text: messageToSend, isComplete: true };
    setChatHistory(prev => [...prev, newUserMessage]);
    if (!messageOverride) {
      setChatInput('');
    }
    setIsChatLoading(true);
    
    const placeholderMessage: NoteChatMessage = { role: 'model', text: '', isComplete: false };
    setChatHistory(prev => [...prev, placeholderMessage]);

    try {
      // Step 1: ALWAYS query the knowledge base
      const backendResponse = await fetch('/api/find-related', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: messageToSend, collection_name: knowledgeBaseId, top_k: 5 })
      });
      if (!backendResponse.ok) {
        const errorText = await backendResponse.text().catch(() => backendResponse.statusText);
        throw new Error(`知识库查询失败: ${errorText}`);
      }
      
      const result = await backendResponse.json();
      if (result.error) throw new Error(`知识库返回错误: ${result.error}`);

      const retrievedSources: Source[] = result.related_documents || [];
      const finalSources: Source[] | undefined = retrievedSources;
      
      const context = retrievedSources.map((s: Source) => `
<document>
  <source>${s.source_file}</source>
  <content>
    ${s.content_chunk}
  </content>
</document>
`).join('');
      
      const systemInstruction = `You are a helpful Q&A assistant. Answer the user's question based ONLY on the provided documents.
- Structure your answer clearly using Markdown formatting (like lists, bold text, etc.).
- For each piece of information or claim in your answer, you MUST cite its origin by appending "[Source: file_name.txt]" at the end of the sentence.
- You must use the exact filename from the <source> tag of the document you used.
- If the information comes from multiple sources, cite them all, like "[Source: file1.txt], [Source: file2.txt]".
- If you cannot answer the question from the documents, state that clearly. Do not use outside knowledge.`;
      
      const userPrompt = `[DOCUMENTS]${context}\n\n[USER QUESTION]\n${messageToSend}`;
      const chatHistoryForApi: ChatMessage[] = []; // No history is passed for KB-mode to force focus on provided context

      await callGenerativeAiStream(
          provider, systemInstruction, userPrompt, apiKeys, chatHistoryForApi,
          (chunk) => {
              setChatHistory(prev => {
                  const newHistory = [...prev];
                  const lastMessage = newHistory[newHistory.length - 1];
                  if (lastMessage?.role === 'model') {
                      lastMessage.text += chunk;
                  }
                  return newHistory;
              });
          },
          () => { // onComplete
               setChatHistory(prev => {
                  const newHistory = [...prev];
                  const lastMessage = newHistory[newHistory.length - 1];
                  if (lastMessage?.role === 'model') {
                      lastMessage.sources = finalSources;
                      lastMessage.isComplete = true;
                  }
                  return newHistory;
              });
              setIsChatLoading(false);
          },
          (error) => { throw error; }
      );

    } catch (error: any) {
        setChatHistory(prev => {
            const newHistory = [...prev];
            const lastMessage = newHistory[newHistory.length - 1];
            if (lastMessage?.role === 'model') {
                lastMessage.text = `抱歉，处理时出错了: ${error.message}`;
                lastMessage.isError = true;
                lastMessage.isComplete = true;
            }
            return newHistory;
        });
        setIsChatLoading(false);
    }
  }, [chatInput, isChatLoading, knowledgeBaseId, provider, apiKeys]);

  useEffect(() => {
    if (initialQuestion && !isInitialQuestionSent.current) {
      isInitialQuestionSent.current = true;
      handleSendChatMessage(undefined, initialQuestion);
    }
  }, [initialQuestion, handleSendChatMessage]);
  
  const handleHistoryClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const target = e.target as HTMLElement;
      if (target.classList.contains('source-citation')) {
          e.preventDefault();
          const filename = target.dataset.filename;
          if (!filename) return;

          const messageElement = target.closest('.kb-message');
          if (!messageElement) return;

          const sourceItem = messageElement.querySelector(`.source-item[data-filename="${filename}"]`) as HTMLLIElement;
          
          if (sourceItem) {
              const details = sourceItem.closest('details');
              if (details && !details.open) {
                  details.open = true;
              }
              
              sourceItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
              
              sourceItem.classList.add('highlighted');
              setTimeout(() => {
                  sourceItem.classList.remove('highlighted');
              }, 2500);
          }
      }
  };

  return (
    <div className="kb-view-container">
        <div className="view-header-row">
          <h2 style={{textTransform: 'capitalize'}}>知识库对话: {knowledgeBaseName} (由 {provider} 模型生成)</h2>
          <button className="btn btn-secondary" onClick={handleExportTXT} disabled={chatHistory.length <= 1}>
            导出 TXT
          </button>
        </div>
        <div className="kb-chat-history" ref={chatHistoryRef} onClick={handleHistoryClick}>
            {chatHistory.map((msg, index) => (
                <div key={index} className={`kb-message ${msg.role} ${msg.isError ? 'error' : ''}`}>
                    <div className="avatar-icon">{msg.role === 'user' ? 'U' : 'M'}</div>
                    <div className="message-content">
                        {(msg.role === 'model' && msg.isComplete)
                            ? <div dangerouslySetInnerHTML={{ __html: parseMessageText(msg.text) }} />
                            : <p>{msg.text}</p>
                        }
                        {msg.sources && msg.sources.length > 0 && (
                            <details className="source-info-box" open>
                                <summary>参考源头信息 ({msg.sources.length})</summary>
                                <ul className="source-list">
                                    {msg.sources.map((source, i) => (
                                        <li key={i} className="source-item" data-filename={source.source_file}>
                                            <div className="source-header">
                                                <span className="source-filename">{source.source_file}</span>
                                                <span className="source-score">Relevance: {source.score.toFixed(2)}</span>
                                            </div>
                                            <p className="source-chunk">"{source.content_chunk}"</p>
                                        </li>
                                    ))}
                                </ul>
                            </details>
                        )}
                    </div>
                </div>
            ))}
             {isChatLoading && chatHistory[chatHistory.length - 1]?.role === 'model' && !chatHistory[chatHistory.length - 1]?.text && (
                <div className="spinner-container" style={{padding: '10px 0'}}><div className="spinner"></div></div>
            )}
        </div>
        <form className="chat-input-form" onSubmit={handleSendChatMessage}>
            <textarea
                className="chat-input"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendChatMessage(); }}}
                placeholder="在此输入您的问题..."
                rows={2}
                disabled={isChatLoading}
            />
            <button type="submit" className="btn btn-primary send-btn" disabled={isChatLoading || !chatInput.trim()}>发送</button>
        </form>
    </div>
  );
};

const DiffView = ({ originalText, revisedText }: { originalText: string; revisedText: string }) => {
    const diff = (a: string[], b: string[]) => {
        const matrix = Array(a.length + 1).fill(null).map(() => Array(b.length + 1).fill(0));
        for (let i = 1; i <= a.length; i++) {
            for (let j = 1; j <= b.length; j++) {
                if (a[i - 1] === b[j - 1]) {
                    matrix[i][j] = matrix[i - 1][j - 1] + 1;
                } else {
                    matrix[i][j] = Math.max(matrix[i - 1][j], matrix[i][j - 1]);
                }
            }
        }
        let i = a.length;
        let j = b.length;
        const result: { value: string; type: 'common' | 'removed' | 'added' }[] = [];
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
                result.unshift({ value: a[i - 1], type: 'common' });
                i--;
                j--;
            } else if (j > 0 && (i === 0 || matrix[i][j - 1] >= matrix[i - 1][j])) {
                result.unshift({ value: b[j - 1], type: 'added' });
                j--;
            } else if (i > 0 && (j === 0 || matrix[i][j - 1] < matrix[i - 1][j])) {
                result.unshift({ value: a[i - 1], type: 'removed' });
                i--;
            } else {
                break;
            }
        }
        return result;
    };

    const diffResult = diff(originalText.split(/(\s+)/), revisedText.split(/(\s+)/));

    return (
        <div className="diff-view">
            {diffResult.map((part, index) => {
                if (part.type === 'added') {
                    return <span key={index} className="diff-add">{part.value}</span>;
                }
                if (part.type === 'removed') {
                    return <span key={index} className="diff-remove">{part.value}</span>;
                }
                return <span key={index}>{part.value}</span>;
            })}
        </div>
    );
};


const WritingView = ({
  initialText,
  onTextChange,
  apiKeys,
  selectedModel,
  selectedKnowledgeBase,
  knowledgeBases,
}: {
  initialText: string;
  onTextChange: (newText: string) => void;
  apiKeys: { [key in ModelProvider]?: string };
  selectedModel: ModelProvider;
  selectedKnowledgeBase: string | null;
  knowledgeBases: { id: string; name: string }[];
}) => {
  const [text, setText] = useState(initialText);
  const [suggestions, setSuggestions] = useState<WritingSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [styleReferenceText, setStyleReferenceText] = useState('');
  const [copySuccess, setCopySuccess] = useState('');

  const [kbResults, setKbResults] = useState<Source[] | null>(null);
  const [isKbSearching, setIsKbSearching] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState<number | null>(null);

  const editorRef = useRef<HTMLTextAreaElement>(null);
  const styleFileRef = useRef<HTMLInputElement>(null);
  const suppressSuggestionFetch = useRef(false);
  const fetchIdRef = useRef(0);

  const fetchSuggestions = useCallback(debounce(async (currentText: string, styleText: string) => {
    if (currentText.trim().length < 50) { // Don't run on very short text
      setSuggestions([]);
      return;
    }
    const fetchId = ++fetchIdRef.current;
    setIsLoading(true);
    setError(null);
    setSelectedSuggestionIndex(null);

    let systemInstruction = `你是一位专业的中文写作助理。你的任务是实时帮助用户改进他们的中文写作。
- 分析所提供的文本，并找出最多3个关键的改进点。
- 保持文档原有的语调和风格。
- 针对每一条建议，提供精准的原文片段 ("originalText")、你修改后的版本 ("revisedText")，以及简明扼要的修改说明 ("explanation")。
- 你所有的输出，包括建议和说明，都必须是中文。
- 你的整个响应必须是一个JSON对象数组，每个对象包含 "originalText"、"revisedText" 和 "explanation" 这三个键。
- 如果文本写得很好，无需修改，请返回一个空数组 []。`;

    if (styleText.trim()) {
        systemInstruction += `\n\n- 重要：你必须严格遵循以下“写作风格参考”文档中的写作风格、语气和词汇。

[写作风格参考]:
---
${styleText.trim()}
---
`;
    }

    const userPrompt = `[Text for Analysis]:\n\n${currentText}`;

    try {
      const responseText = await callGenerativeAi(selectedModel, systemInstruction, userPrompt, true, apiKeys, 'writing');
      
      // If another request has started, ignore the result of this one to prevent race conditions.
      if (fetchId !== fetchIdRef.current) return;

      const { data, error: parseError, rawResponse } = parseJsonResponse<WritingSuggestion[]>(responseText);

      if (parseError || !data) {
        console.error("Raw response on parse error:", rawResponse);
        throw new Error(parseError || "Received invalid data from model.");
      }
      
      const validSuggestions = data.filter(s => s.originalText && s.revisedText && s.explanation);
      setSuggestions(validSuggestions);

    } catch (err: any) {
      if (fetchId === fetchIdRef.current) {
          setError(`获取建议失败: ${err.message}`);
          setSuggestions([]);
      }
    } finally {
      if (fetchId === fetchIdRef.current) {
          setIsLoading(false);
      }
    }
  }, 1500), [selectedModel, apiKeys]);

  useEffect(() => {
    onTextChange(text);
    if (suppressSuggestionFetch.current) {
        suppressSuggestionFetch.current = false; // Reset for next non-apply update
        return;
    }
    fetchSuggestions(text, styleReferenceText);
  }, [text, styleReferenceText, fetchSuggestions, onTextChange]);

  const handleApplySuggestion = (suggestion: WritingSuggestion) => {
    suppressSuggestionFetch.current = true;
    setText(prevText => prevText.replace(suggestion.originalText, suggestion.revisedText));
    setSuggestions(prev => prev.filter(s => s !== suggestion));
    setSelectedSuggestionIndex(null); // Deselect after applying
  };
  
  const handleRefresh = () => {
      fetchSuggestions.clearTimeout(); // Cancel any pending debounced call
      fetchSuggestions(text, styleReferenceText);
  };
  
  const handleSuggestionClick = (suggestion: WritingSuggestion, index: number) => {
    setSelectedSuggestionIndex(index);
    if (editorRef.current) {
        const fullText = editorRef.current.value;
        const startIndex = fullText.indexOf(suggestion.originalText);
        if (startIndex !== -1) {
            const endIndex = startIndex + suggestion.originalText.length;
            editorRef.current.focus();
            editorRef.current.setSelectionRange(startIndex, endIndex);
        }
    }
  };

  const handleCopy = () => {
    if (text && navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            setCopySuccess('已复制!');
            setTimeout(() => setCopySuccess(''), 2000);
        }).catch(err => {
            setCopySuccess('复制失败!');
            setTimeout(() => setCopySuccess(''), 2000);
            console.error('Failed to copy text: ', err)
        });
    }
  };
  
  const handleKbSearch = async () => {
    if (!selectedKnowledgeBase) {
        setKbError("请返回首页选择一个知识库。");
        return;
    }
    if (text.trim().length < 20) {
        setKbError("请写入更多内容以便进行有效检索。");
        return;
    }

    setIsKbSearching(true);
    setKbError(null);
    setKbResults(null);

    try {
        const backendResponse = await fetch('/api/find-related', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                collection_name: selectedKnowledgeBase,
                top_k: 3
            })
        });
        if (!backendResponse.ok) {
            const errorText = await backendResponse.text();
            throw new Error(errorText || "知识库查询失败。");
        }
        const data = await backendResponse.json();
        if (data.error) {
            throw new Error(data.error);
        }
        const sources = data.related_documents || [];
        setKbResults(sources);
        if (sources.length === 0) {
            setKbError("未找到相关内容。");
        }

    } catch (err: any) {
        setKbError(`知识库检索出错: ${err.message}`);
    } finally {
        setIsKbSearching(false);
    }
  };

  const processStyleFile = async (file: File) => {
    if (!file) return;
    const reader = new FileReader();

    reader.onload = async (event) => {
        const fileContent = event.target?.result;
        let fileText = '';
        if (file.name.endsWith('.docx')) {
            try {
                const result = await mammoth.extractRawText({ arrayBuffer: fileContent as ArrayBuffer });
                fileText = result.value;
            } catch (err) {
                console.error("Error reading docx file", err);
                setError("无法解析 DOCX 文件。");
                return;
            }
        } else {
            fileText = fileContent as string;
        }
        setStyleReferenceText(fileText);
    };
    
    if (file.name.endsWith('.docx')) {
        reader.readAsArrayBuffer(file);
    } else if (file.name.endsWith('.txt') || file.name.endsWith('.md')) {
        reader.readAsText(file);
    } else {
        alert("不支持的文件类型。请上传 .txt, .md 或 .docx 文件。");
    }
  };

  const handleStyleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
        processStyleFile(e.target.files[0]);
        e.target.value = '';
    }
  };

  const handleUploadStyleClick = () => {
      styleFileRef.current?.click();
  };

  return (
    <div className="writing-view-container">
      <div className="writing-editor-panel">
         <div className="assistant-panel-header">
            <h2>沉浸式写作</h2>
            <button className="btn btn-secondary" onClick={handleCopy} disabled={!text}>
                 {copySuccess || '复制全文'}
            </button>
        </div>
        <textarea
          ref={editorRef}
          className="text-area writing-editor"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="在此开始写作，AI 将在您停顿时提供建议..."
        />
      </div>
      <div className="writing-assistant-panel">
        <div className="assistant-panel-header">
            <h2 style={{textTransform: 'capitalize'}}>AI 助手 ({selectedModel})</h2>
            <button className="btn btn-secondary" onClick={handleRefresh} disabled={isLoading}>
                {isLoading ? <span className="spinner"></span> : null}
                刷新建议
            </button>
        </div>
        <div className="assistant-content">
          {isLoading && <div className="spinner-container"><div className="spinner large" /></div>}
          {!isLoading && error && <div className="error-message">{error}</div>}
          {!isLoading && !error && suggestions.length === 0 && (
            <div className="large-placeholder">
              <p>暂无建议。</p>
              <p className="instruction-text">请继续写作，或确保文本长度超过50个字符以便AI分析。</p>
            </div>
          )}
          {!isLoading && !error && suggestions.length > 0 && (
            <div className="suggestions-list">
              {suggestions.map((s, i) => (
                <div 
                  key={i} 
                  className={`suggestion-card ${selectedSuggestionIndex === i ? 'selected' : ''}`}
                  onClick={() => handleSuggestionClick(s, i)}
                >
                  <div className="suggestion-body">
                    <p><strong>差异对比:</strong></p>
                    <DiffView originalText={s.originalText} revisedText={s.revisedText} />
                    <p style={{marginTop: '8px'}}><strong>说明:</strong> {s.explanation}</p>
                  </div>
                  <div className="suggestion-actions">
                    <button className="btn btn-primary" onClick={(e) => { e.stopPropagation(); handleApplySuggestion(s); }}>
                      应用此建议
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="style-reference-section">
            <h4>写作风格参考 (可选)</h4>
            <textarea
                className="text-area style-reference-textarea"
                value={styleReferenceText}
                onChange={(e) => setStyleReferenceText(e.target.value)}
                placeholder="在此处粘贴范文，或上传文件，以固定AI的写作风格..."
            />
             <input
                type="file"
                ref={styleFileRef}
                style={{ display: 'none' }}
                accept=".txt,.md,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                onChange={handleStyleFileChange}
            />
            <div className="utility-btn-group" style={{justifyContent: 'flex-end'}}>
                 <button className="btn btn-secondary" onClick={handleUploadStyleClick}>
                    上传风格文件
                </button>
                <button className="btn btn-secondary" onClick={() => setStyleReferenceText('')} disabled={!styleReferenceText}>
                    清空风格参考
                </button>
            </div>
        </div>
        <div className="kb-search-section">
            <h4>知识库探索</h4>
            <button className="btn btn-secondary" onClick={handleKbSearch} disabled={isKbSearching || !selectedKnowledgeBase}>
                {isKbSearching ? <span className="spinner"></span> : null}
                {isKbSearching ? '检索中...' : '检索关联内容'}
            </button>
            {kbError && <div className="error-message" style={{textAlign: 'left'}}>{kbError}</div>}
            {kbResults && (
                <div className="source-info-box" style={{marginTop: '8px'}}>
                    <ul className="source-list" style={{maxHeight: '150px', padding: '12px'}}>
                        {kbResults.map((source: Source, i: number) => (
                            <li key={i} className="source-item" data-filename={source.source_file}>
                                <div className="source-header">
                                    <span className="source-filename">{source.source_file}</span>
                                    <span className="source-score">Relevance: {source.score.toFixed(2)}</span>
                                </div>
                                <p className="source-chunk">"{source.content_chunk}"</p>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
      </div>
    </div>
  );
};


const App = () => {
    type View = 'home' | 'notes' | 'audit' | 'knowledge-chat' | 'writing';
    const [view, setView] = useState<View>('home');
    const [inputText, setInputText] = useState('');
    const [noteAnalysisResult, setNoteAnalysisResult] = useState<NoteAnalysis | null>(null);
    const [noteAnalysisError, setNoteAnalysisError] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isApiKeyModalOpen, setIsApiKeyModalOpen] = useState(false);
    const [isThoughtsModalOpen, setIsThoughtsModalOpen] = useState(false);
    
    const [apiKeys, setApiKeys] = useState<{ [key in ModelProvider]?: string }>(() => {
        try {
            const savedKeys = localStorage.getItem('apiKeys');
            const parsedKeys = savedKeys ? JSON.parse(savedKeys) : {};
            // Set keys from environment variables if available
            parsedKeys.openai = parsedKeys.openai || process.env.OPENAI_API_KEY;
            parsedKeys.deepseek = parsedKeys.deepseek || process.env.DEEPSEEK_API_KEY;
            parsedKeys.ali = parsedKeys.ali || process.env.ALI_API_KEY;
            return parsedKeys;
        } catch (e) {
            console.error("Failed to parse API keys from localStorage", e);
            return {};
        }
    });

    const [selectedModel, setSelectedModel] = useState<ModelProvider>('gemini');

    const [knowledgeBases, setKnowledgeBases] = useState<{ id: string; name: string }[]>([]);
    const [isKbLoading, setIsKbLoading] = useState(true);
    const [kbError, setKbError] = useState<string | null>(null);
    const [selectedKnowledgeBase, setSelectedKnowledgeBase] = useState<string | null>(null);
    const [initialKnowledgeChatQuestion, setInitialKnowledgeChatQuestion] = useState<string | undefined>();
    
    useEffect(() => {
        const fetchKnowledgeBases = async () => {
            // Check if running in production (e.g., on GitHub Pages)
            // Vite sets `process.env.NODE_ENV` to 'production' for production builds.
            if (process.env.NODE_ENV === 'production') {
                setKbError("知识库功能在此在线演示版中不可用。此功能需要连接本地后端服务才能运行。");
                setIsKbLoading(false);
                setKnowledgeBases([]);
                setSelectedKnowledgeBase(null);
                return;
            }

            setIsKbLoading(true);
            setKbError(null);
            try {
                const response = await fetch('/api/list-collections');
                if (!response.ok) {
                    const errorText = await response.text().catch(() => response.statusText);
                    let errorJson;
                    try { errorJson = JSON.parse(errorText); } catch(e) {/* ignored */}
                    throw new Error(errorJson?.error || `获取知识库列表失败 (状态: ${response.status})`);
                }
                const data = await response.json();
                const collections: string[] = data.collections || [];
                const formattedKbs = collections.map(name => ({ id: name, name }));
                setKnowledgeBases(formattedKbs);
                // If nothing is selected, or the previously selected one no longer exists, select the first one.
                if (formattedKbs.length > 0) {
                    if (!selectedKnowledgeBase || !collections.includes(selectedKnowledgeBase)) {
                       setSelectedKnowledgeBase(formattedKbs[0].id);
                    }
                } else {
                    setSelectedKnowledgeBase(null);
                }
            } catch (error: any) {
                console.error("Failed to fetch knowledge bases:", error);
                const userFriendlyError = "无法连接到知识库服务。请确保本地后端服务正在运行，并刷新页面重试。";
                setKbError(userFriendlyError);
                setKnowledgeBases([]);
                setSelectedKnowledgeBase(null);
            } finally {
                setIsKbLoading(false);
            }
        };
        fetchKnowledgeBases();
    }, []); // Run only once on component mount


    const handleAnalysis = async (userThoughts: string) => {
        setIsProcessing(true);
        setNoteAnalysisError(null);
        setNoteAnalysisResult(null);
        setView('notes');

        const systemInstruction = `You are a note organization expert. Structure the user's fragmented notes into a coherent, organized document. Also, analyze and summarize the user's separate "thoughts" about the notes, maintaining them as a distinct section. Your response must be in JSON format with two keys: "organizedText" for the structured notes, and "userThoughts" for the processed user ideas.`;
        const userPrompt = `Here are my notes:\n\n${inputText}\n\nHere are my thoughts on these notes:\n\n${userThoughts}`;

        try {
            const responseText = await callGenerativeAi(selectedModel, systemInstruction, userPrompt, true, apiKeys, 'notes');
            let result;
            try {
                result = JSON.parse(responseText);
            } catch (e: any) {
                console.error('Error parsing note analysis from AI:', responseText);
                throw new Error(`Failed to parse note analysis response: ${e.message}`);
            }
            setNoteAnalysisResult(result);
        } catch (err: any) {
            setNoteAnalysisError(`笔记整理失败: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    };
    
    const handleTriggerOrganize = () => {
        setIsThoughtsModalOpen(true);
    };

    const handleTriggerAudit = () => {
        setView('audit');
    };
    
    const handleTriggerWriting = () => {
        setView('writing');
    };

    const handleCloseThoughtsModal = () => {
        setIsThoughtsModalOpen(false);
    }
    
    const handleSubmitThoughts = (thoughts: string) => {
        setIsThoughtsModalOpen(false);
        handleAnalysis(thoughts);
    }

    const handleTriggerKnowledgeChat = () => {
        if (!selectedKnowledgeBase) {
            alert("请先选择一个知识库。");
            return;
        }
        if (!inputText.trim()) {
            alert("请在工作区输入您的问题。");
            return;
        }
        setInitialKnowledgeChatQuestion(inputText);
        setView('knowledge-chat');
    };

    const handleSaveApiKeys = (keys: { [key in ModelProvider]?: string }) => {
        setApiKeys(keys);
        localStorage.setItem('apiKeys', JSON.stringify(keys));
        setIsApiKeyModalOpen(false);
    };

    const renderView = () => {
        switch (view) {
            case 'notes':
                return <NoteAnalysisView 
                    isLoading={isProcessing} 
                    error={noteAnalysisError} 
                    analysisResult={noteAnalysisResult} 
                    provider={selectedModel}
                    originalText={inputText}
                    apiKeys={apiKeys}
                    selectedKnowledgeBaseId={selectedKnowledgeBase}
                    knowledgeBases={knowledgeBases}
                />;
            case 'audit':
                 return <AuditView initialText={inputText} apiKeys={apiKeys} selectedModel={selectedModel} />;
            case 'knowledge-chat':
                if (!selectedKnowledgeBase) {
                    return <div className="error-message">错误：知识库未选择。请返回首页选择一个知识库。</div>;
                }
                return <KnowledgeChatView
                    knowledgeBaseId={selectedKnowledgeBase}
                    knowledgeBaseName={knowledgeBases.find(kb => kb.id === selectedKnowledgeBase)?.name || selectedKnowledgeBase}
                    initialQuestion={initialKnowledgeChatQuestion}
                    provider={selectedModel}
                    apiKeys={apiKeys}
                />;
            case 'writing':
                return <WritingView
                    initialText={inputText}
                    onTextChange={setInputText}
                    apiKeys={apiKeys}
                    selectedModel={selectedModel}
                    selectedKnowledgeBase={selectedKnowledgeBase}
                    knowledgeBases={knowledgeBases}
                />;
            case 'home':
            default:
                return (
                    <HomeInputView 
                        inputText={inputText}
                        setInputText={setInputText}
                        onOrganize={handleTriggerOrganize}
                        onAudit={handleTriggerAudit}
                        selectedModel={selectedModel}
                        setSelectedModel={setSelectedModel}
                        onOpenApiModal={() => setIsApiKeyModalOpen(true)}
                        apiKeys={apiKeys}
                        isProcessing={isProcessing}
                        knowledgeBases={knowledgeBases}
                        isKbLoading={isKbLoading}
                        kbError={kbError}
                        selectedKnowledgeBase={selectedKnowledgeBase}
                        setSelectedKnowledgeBase={setSelectedKnowledgeBase}
                        onKnowledgeChat={handleTriggerKnowledgeChat}
                        onWriting={handleTriggerWriting}
                    />
                );
        }
    };

    const handleBackToHome = () => {
        setNoteAnalysisResult(null);
        setNoteAnalysisError(null);
        // Do not reset selectedKnowledgeBase, so it can be used again
        setInitialKnowledgeChatQuestion(undefined);
        setView('home');
    }

    return (
        <div className="main-layout">
            <div className="app-header">
                <h1>写作笔记助手</h1>
                 <div className="button-group">
                    {view !== 'home' && <button className="btn btn-secondary" onClick={handleBackToHome}>返回首页</button>}
                 </div>
            </div>

            <div className="view-container">
              {renderView()}
            </div>
            
            <ApiKeyModal 
                isOpen={isApiKeyModalOpen}
                onClose={() => setIsApiKeyModalOpen(false)}
                onSave={handleSaveApiKeys}
                apiKeys={apiKeys}
            />
            <ThoughtsInputModal
                isOpen={isThoughtsModalOpen}
                onClose={handleCloseThoughtsModal}
                onSubmit={handleSubmitThoughts}
             />
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);