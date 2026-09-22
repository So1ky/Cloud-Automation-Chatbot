# apply 이후 생성된 리소스의 식별자를 터미널에 출력한다. 단계별 검증에 쓴다.
output "vpc_id" {
  description = "생성된 VPC의 ID"
  value       = aws_vpc.main.id
}

output "public_subnet_id" {
  description = "퍼블릭 서브넷의 ID"
  value       = aws_subnet.public.id
}

output "node_security_group_id" {
  description = "노드 보안그룹 ID"
  value       = aws_security_group.node.id
}

output "key_pair_name" {
  description = "EC2 키페어 이름 (비상 접속용)"
  value       = aws_key_pair.main.key_name
}

output "instance_profile_name" {
  description = "EC2에 붙일 Instance Profile 이름"
  value       = aws_iam_instance_profile.node.name
}

output "ecr_repository_urls" {
  description = "이미지를 push할 ECR 리포지토리 URL (docker tag 대상)"
  value       = { for k, v in aws_ecr_repository.app : k => v.repository_url }
}

output "node_private_ips" {
  description = "노드 프라이빗 IP (stop/start해도 유지된다)"
  value       = { for k, v in aws_instance.node : k => v.private_ip }
}

output "node_public_ips" {
  description = "노드 공인 IP (기동할 때마다 바뀐다 — up.sh가 DNS를 갱신한다)"
  value       = { for k, v in aws_instance.node : k => v.public_ip }
}

output "node_instance_ids" {
  description = "SSM 접속에 쓰는 인스턴스 ID"
  value       = { for k, v in aws_instance.node : k => v.id }
}

output "hosted_zone_id" {
  description = "호스팅 존 ID. up.sh의 A레코드 갱신(change-resource-record-sets)에 쓴다"
  value       = aws_route53_zone.main.zone_id
}

output "name_servers" {
  description = "존의 NS. 도메인 등록 정보의 NS와 일치해야 DNS가 동작한다"
  value       = aws_route53_zone.main.name_servers
}

output "efs_id" {
  description = "rag-assets EFS 파일시스템 ID. k8s static PV의 nfs server 주소 구성에 쓴다"
  value       = aws_efs_file_system.rag_assets.id
}

output "efs_dns_name" {
  description = "EFS 마운트용 DNS 이름 (<fs-id>.efs.<region>.amazonaws.com)"
  value       = aws_efs_file_system.rag_assets.dns_name
}
