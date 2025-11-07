// This file is now an ambient declaration file (a script file), not a module.
// This allows `declare module 'mammoth'` to be a primary declaration, fixing the `export =` error,
// and global interfaces are declared directly without `declare global`.

// FIX: Wrapped global augmentations in `declare global` to correctly modify the global scope
// when the file is treated as a module, resolving TypeScript errors.
declare global {
  // These interfaces will be merged into the global scope automatically by TypeScript,
  // augmenting Vite's built-in types without conflict.
  interface ImportMetaEnv {
    // FIX: Removed PROD and DEV as they are already defined by Vite's client types,
    // resolving the "All declarations of '...' must have identical modifiers" error.
    readonly VITE_API_BASE_URL?: string;
    readonly VITE_GEMINI_API_KEY?: string;
    readonly VITE_OPENAI_API_KEY?: string;
    readonly VITE_OPENAI_TARGET_URL?: string;
    readonly VITE_OPENAI_MODEL?: string;
    readonly VITE_DEEPSEEK_API_KEY?: string;
    readonly VITE_DEEPSEEK_ENDPOINT?: string;
    readonly VITE_DEEPSEEK_MODEL?: string;
    readonly VITE_ALI_API_KEY?: string;
    readonly VITE_ALI_TARGET_URL?: string;
    readonly VITE_ALI_ENDPOINT?: string;
    readonly VITE_ALI_MODEL?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }

  // Augment the existing NodeJS.ProcessEnv interface for environment variables.
  namespace NodeJS {
    interface ProcessEnv {
      readonly API_KEY?: string;
    }
  }
}

// Declare the 'mammoth' module for CommonJS interoperability.
declare module 'mammoth' {
  interface MammothResult {
    value: string;
    messages: any[];
  }

  const mammoth: {
    extractRawText(options: { arrayBuffer: ArrayBuffer }): Promise<MammothResult>;
  };

  // FIX: `export =` is now valid because this `declare module` block is in a global
  // script file, making it a primary module declaration, not an augmentation.
  export = mammoth;
}
