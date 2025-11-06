// FIX: Add Vite client types reference to provide type definitions for `import.meta.env` and resolve TypeScript errors.
/// <reference types="vite/client" />

declare namespace NodeJS {
  interface ProcessEnv {
    NODE_ENV: 'development' | 'production';
    API_KEY?: string;
    OPENAI_API_KEY?: string;
    DEEPSEEK_API_KEY?: string;
    ALI_API_KEY?: string;
    OPENAI_ENDPOINT?: string;
    OPENAI_MODEL?: string;
    DEEPSEEK_ENDPOINT?: string;
    DEEPSEEK_MODEL?: string;
    ALI_ENDPOINT?: string;
    ALI_MODEL?: string;
  }
}

declare module 'mammoth' {
  interface MammothResult {
    value: string;
    messages: any[];
  }

  const mammoth: {
    extractRawText(options: { arrayBuffer: ArrayBuffer }): Promise<MammothResult>;
  };

  export default mammoth;
}
