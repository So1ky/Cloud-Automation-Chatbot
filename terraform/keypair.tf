# 비상용 SSH 키페어.
# 평상시 접속은 SSM Session Manager를 쓰고 22번 포트는 열지 않는다.
# 다만 ssm-agent가 기동하지 않는 상황을 대비해 키페어 자체는 만들어 둔다 —
# 키페어는 EC2 생성 시점에만 지정할 수 있어 나중에 추가할 수 없다.
# (비상시에는 보안그룹에 22번 규칙을 임시로 추가한다)
resource "aws_key_pair" "main" {
  key_name   = "${var.project_name}-key"
  public_key = file(pathexpand(var.ssh_public_key_path))

  tags = {
    Name = "${var.project_name}-key"
  }
}
