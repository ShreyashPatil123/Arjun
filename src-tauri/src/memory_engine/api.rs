//! The legacy memory IPC surface, removed.
//!
//! ## What was here
//!
//! Ten Tauri commands: `get_memory_health_status`, `get_user_profile_memory`,
//! `update_user_profile_fact`, `list_memory_projects`, `create_memory_project`,
//! `switch_active_project`, `get_active_project`, `search_memory_nodes`,
//! `delete_memory_node_by_id` and `get_memory_diagnostics`.
//!
//! ## Why they went
//!
//! Every one of them was written the same way:
//!
//! ```text
//! require_session(&session)?;
//! memory_mgr.get_user_profile()
//! ```
//!
//! The first line proves that *somebody* is signed in. The second takes no user
//! id, because nothing below it had one: the profile table, the project table,
//! the memory nodes, the summaries and the active-project selection were all
//! per-machine, not per-person. So any signed-in user could list, search,
//! update, switch and delete every other user's memory, and the comment above
//! the health command said "the per-user scoping lives inside the manager" —
//! which was not true of any of them.
//!
//! Two ways out were available: retrofit `user_id` ownership onto every table
//! with a migration, or remove the surface. Removal was correct here because
//! the surface had no consumer at all. `src/services/memoryService.ts` wrapped
//! all ten and was imported by no component; a `grep` for it across `src/`
//! returned only its own definition. Retrofitting ownership onto an API nobody
//! called would have been building a lock for a door onto a field.
//!
//! ## What memory a run actually uses
//!
//! [`crate::agent_runtime::memory::MemoryStore`], reached only through
//! [`crate::agent_runtime::memory_api`], which fills in identity, project,
//! classification and approval on the Rust side rather than taking them from
//! the caller. Its boundary is tested in
//! `crate::agent_runtime::memory_boundary_tests`.
//!
//! ## What remains of this module
//!
//! [`super::MemoryManager`] is still constructed, and still used by one
//! command: `send_chat_message` in `crate::commands::inference`, the SDK chat
//! path. It is an internal detail of that path and is no longer reachable over
//! IPC. Consolidating that path onto `MemoryStore` is the remaining work; until
//! it happens, nothing here is exposed to a caller.
