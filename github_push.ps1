# github_push.ps1 — 用 Personal Access Token 创建并推送 GitHub 仓库
#
# GitHub 自 2021-08 起不再接受账号密码认证, 必须使用 PAT。
# 使用方法:
#   1) 生成 PAT: GitHub → Settings → Developer settings → Personal access tokens
#      → Tokens (classic) → Generate new token, 勾选 repo 权限
#   2) 运行:
#        $env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxx"
#        .\github_push.ps1
#    或:
#        .\github_push.ps1 -Token ghp_xxxxxxxxxxxxxxxx
param(
  [string]$RepoName = "Simple-simulation",
  [string]$Description = "nsjy: Abelian surface / lattice / moduli (t1,t2) pipeline — C# + Python algorithm suite",
  [string]$Token = $env:GITHUB_TOKEN
)

if (-not $Token) {
  Write-Error "需要 GITHUB_TOKEN (Personal Access Token)。见脚本头部说明。"
  exit 1
}

$headers = @{ Authorization = "Bearer $Token" }

# 1. 获取认证用户名
$user = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers
Write-Host "认证用户: $($user.login)"

# 2. 创建仓库 (已存在则跳过)
try {
  $body = @{ name = $RepoName; description = $Description; private = $false } | ConvertTo-Json
  Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Headers $headers -Method Post `
    -Body $body -ContentType "application/json" | Out-Null
  Write-Host "仓库已创建: https://github.com/$($user.login)/$RepoName"
} catch {
  if ($_.ErrorDetails.Message -match "already exists") {
    Write-Host "仓库已存在: https://github.com/$($user.login)/$RepoName"
  } else {
    throw
  }
}

# 3. 设置 remote 并推送 (token 作为一次性认证, 之后清理)
$remote = "https://x-access-token:$Token@github.com/$($user.login)/$RepoName.git"
git remote remove origin 2>$null
git remote add origin $remote
git branch -M main
git push -u origin main
if ($LASTEXITCODE -ne 0) { Write-Error "push 失败"; exit 1 }

# 4. 清理: 换回不含 token 的干净 URL (token 不落盘)
git remote set-url origin "https://github.com/$($user.login)/$RepoName.git"
Write-Host "完成: https://github.com/$($user.login)/$RepoName"
