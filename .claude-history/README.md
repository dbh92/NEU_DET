# Lịch sử hội thoại Claude Code

Thư mục này chứa bản sao lịch sử hội thoại Claude Code của dự án (`sessions/*.jsonl`),
để clone về máy khác vẫn xem lại và tiếp tục được.

> ⚠️ File `.jsonl` chứa **toàn bộ** hội thoại: nội dung trao đổi, output lệnh, đường dẫn máy,
> email trong git config. Chỉ nên để trong repo **private**.

## Máy mới: khôi phục (chạy 1 lần sau khi clone)

```powershell
powershell -ExecutionPolicy Bypass -File .claude-history\restore.ps1
```

Sau đó mở thư mục bằng VS Code → Claude Code → danh sách hội thoại cũ
(hoặc chạy `claude --resume` trong terminal).

Script tự tính tên thư mục trong `~\.claude\projects\` từ đường dẫn clone hiện tại,
nên clone ở đâu cũng được. Nếu máy đó đã có bản mới hơn, script sẽ bỏ qua, không ghi đè.

## Trước mỗi lần commit: lưu lịch sử mới nhất vào repo

```powershell
powershell -ExecutionPolicy Bypass -File .claude-history\save.ps1
git add .claude-history
```

## Không có lịch sử vẫn tiếp tục được

[CLAUDE.md](../CLAUDE.md) ở gốc dự án chứa tiến độ, quy ước và bước đang làm. Claude Code
tự đọc file này, nên mở một hội thoại **mới** cũng tiếp tục đúng chỗ đang dừng.
