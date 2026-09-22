# VPC — 계정 안에 만드는, 논리적으로 격리된 사설 네트워크.
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"

  # VPC 내부에서 AWS 서비스 도메인(ECR 등)을 이름으로 해석하려면 필요하다.
  enable_dns_support = true
  # 공인 IP를 가진 인스턴스에 DNS 호스트명을 부여한다.
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# 서브넷 — VPC 대역을 잘라 가용영역 하나에 배치한 구간. 인스턴스는 서브넷에만 놓인다.
resource "aws_subnet" "public" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a" # ap-northeast-2a

  # 이 서브넷에 뜨는 인스턴스에 공인 IP를 자동 할당한다.
  # EIP(고정 IP)를 쓰지 않기로 했으므로(꺼둔 동안에도 과금) 이 옵션이 필수다.
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-a"
  }
}

# 인터넷 게이트웨이 — VPC와 인터넷을 잇는 관문. VPC에 붙이기만 할 뿐, 경로는 별도다.
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

# 라우팅 테이블 — "어느 목적지를 어디로 보낼지"의 규칙 모음.
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0" # VPC 밖의 모든 목적지를
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

# 라우팅 테이블을 서브넷에 연결한다.
# 이 연결이 있어야 비로소 해당 서브넷이 "퍼블릭 서브넷"이 된다.
resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}
