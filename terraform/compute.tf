locals {
  # 노드별 프라이빗 IP와 루트 디스크 크기.
  #
  # 프라이빗 IP를 고정하는 이유: EC2의 프라이빗 IP는 stop/start해도 유지된다.
  # kubeadm은 이 IP를 기준으로 인증서·etcd·kubelet을 구성하므로,
  # 껐다 켜는 운영에서 클러스터가 스스로를 다시 찾을 수 있다.
  #
  # 디스크를 차등 배분하는 이유: control-plane은 워크로드를 받지 않고,
  # 워커는 backend 이미지(2.93GB)를 받아둘 공간이 필요하다. EBS는 꺼도 과금된다.
  nodes = {
    "control-plane" = { private_ip = "10.0.1.10", disk_gb = 15 }
    "worker-1"      = { private_ip = "10.0.1.11", disk_gb = 25 }
    "worker-2"      = { private_ip = "10.0.1.12", disk_gb = 25 }
  }
}

resource "aws_instance" "node" {
  for_each = local.nodes

  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.public.id
  private_ip    = each.value.private_ip

  vpc_security_group_ids = [aws_security_group.node.id]
  iam_instance_profile   = aws_iam_instance_profile.node.name
  key_name               = aws_key_pair.main.key_name # 비상용. 평상시 접속은 SSM

  root_block_device {
    volume_type = "gp3"
    volume_size = each.value.disk_gb
    encrypted   = true # 기본 KMS 키로 암호화. 추가 비용 없음

    tags = {
      Name = "${var.project_name}-${each.key}-root"
    }
  }

  # 메타데이터 서비스를 IMDSv2만 허용한다.
  # IMDSv1은 토큰 없이 조회되어 SSRF 취약점으로 임시 자격증명이 유출되는 경로가 된다.
  metadata_options {
    http_tokens = "required"
    # 파드(오버레이 네트워크)에서 IMDS에 닿으려면 응답 TTL이 2홉을 견뎌야 한다.
    # 기본값 1이면 EBS CSI 컨트롤러가 노드 IAM 자격증명을 읽지 못한다.
    http_put_response_hop_limit = 2
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    k8s_minor = var.k8s_minor_version
  })

  # user_data는 최초 부팅 시 1회만 실행된다.
  # false로 두면 스크립트를 고쳐도 이미 떠 있는 인스턴스는 교체되지 않는다 —
  # 클러스터를 세운 뒤 실수로 스크립트를 건드려 서버가 날아가는 것을 막기 위함이다.
  # 초기 구축 중 스크립트를 수정했다면 명시적으로 재생성한다:
  #   terraform apply -replace='aws_instance.node["control-plane"]'
  user_data_replace_on_change = false

  tags = {
    Name = "${var.project_name}-${each.key}"
    Role = each.key
  }
}
