/**
 * Re-export of the shared `useConversation` hook.
 *
 * The hook itself lives in `src/contexts/ConversationContext.tsx` so the
 * app shell, the chat surface, the composer, the task panel, and the
 * context chip all read the SAME state instead of each having their
 * own `useState` slice. See the context file for the per-run reducer
 * isolation contract.
 */
export {
  useConversation,
  type UseConversation,
  ConversationProvider,
  type MessageEvent,
  isMessageEvent,
  RunReducer,
  RunReducerRegistry,
} from '../../contexts/ConversationContext';
