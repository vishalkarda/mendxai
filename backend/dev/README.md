Template and workflow for `dev` change tracking

Purpose
- Use `template_change.log` as a per-change entry to record intent, scope, and verification steps.

How to use
1. Copy `template_change.log` to a new file named `change_<n>_YYYYMMDD.log` or create a PR and add the completed template as part of the change commit.
2. Fill `Change-ID` with a PR or internal ID and link the related PR/Issue under `Related PR / Issue`.
3. Add verification steps and a short `Rollback Steps` section.

Recommendations
- Keep change files small and focused per batch of related edits.
- Reference the change file in the PR description and in the `dev` folder index.
- Keep `dev/change_*.md` for historical records; new changes should follow the template.
