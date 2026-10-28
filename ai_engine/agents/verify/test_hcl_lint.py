"""hcl_lint 규칙별 통과/실패 테스트 (LLM 호출 없음)."""
import unittest

from ai_engine.agents.verify.hcl_lint import lint_hcl


def _run(main_tf: str):
    return lint_hcl({"main.tf": main_tf})


class RdsTests(unittest.TestCase):
    def test_missing_encryption_and_public(self):
        issues = _run('resource "aws_db_instance" "db" {\n  engine = "postgres"\n  publicly_accessible = true\n}\n')
        self.assertEqual([i["severity"] for i in issues], ["critical", "critical"])
        self.assertIn("storage_encrypted", issues[0]["description"])
        self.assertIn("publicly_accessible", issues[1]["description"])

    def test_ok(self):
        self.assertEqual(_run('resource "aws_db_instance" "db" {\n  storage_encrypted = true\n  publicly_accessible = false\n}\n'), [])


CF = '''
resource "aws_s3_bucket" "videos" {
  bucket = "v"
}
resource "aws_s3_bucket_server_side_encryption_configuration" "videos" {
  bucket = aws_s3_bucket.videos.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.video.arn
    }
  }
}
resource "aws_kms_key" "video" {
  policy = jsonencode({ Statement = [{ Principal = { AWS = "arn:aws:iam::123:root" }, Action = "kms:*" }] })
}
resource "aws_cloudfront_distribution" "cdn" {
  origin {
    domain_name              = aws_s3_bucket.videos.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.oac.id
  }
}
'''


class CloudFrontTests(unittest.TestCase):
    def test_missing_bucket_policy_and_kms_principal(self):
        issues = _run(CF)
        descs = " ".join(i["description"] for i in issues)
        self.assertEqual(len(issues), 2)
        self.assertIn("aws_s3_bucket_policy", descs)
        self.assertIn("cloudfront.amazonaws.com", descs)

    def test_ok_with_policy_and_key_policy(self):
        ok = CF + '''
resource "aws_s3_bucket_policy" "videos" {
  policy = jsonencode({ Statement = [{ Principal = { Service = "cloudfront.amazonaws.com" } }] })
}
resource "aws_kms_key_policy" "video" {
  policy = jsonencode({ Statement = [{ Principal = { Service = "cloudfront.amazonaws.com" }, Action = "kms:Decrypt" }] })
}
'''
        self.assertEqual(_run(ok), [])

    def test_policies_in_data_documents_are_resolved(self):
        ok = CF.replace(
            'policy = jsonencode({ Statement = [{ Principal = { AWS = "arn:aws:iam::123:root" }, Action = "kms:*" }] })',
            "policy = data.aws_iam_policy_document.kms.json") + '''
data "aws_iam_policy_document" "kms" {
  statement {
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    actions = ["kms:Decrypt"]
  }
}
data "aws_iam_policy_document" "bucket" {
  statement {
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
  }
}
resource "aws_s3_bucket_policy" "videos" {
  policy = data.aws_iam_policy_document.bucket.json
}
'''
        self.assertEqual(_run(ok), [])

    def test_unencrypted_origin_needs_only_bucket_policy(self):
        text = CF.replace("kms_master_key_id = aws_kms_key.video.arn", "") + '''
resource "aws_s3_bucket_policy" "videos" {
  policy = jsonencode({ Statement = [{ Principal = { Service = "cloudfront.amazonaws.com" } }] })
}
'''
        self.assertEqual(_run(text), [])


class SqsLambdaTests(unittest.TestCase):
    def test_visibility_shorter_than_timeout(self):
        text = '''
resource "aws_sqs_queue" "q" {
  name = "q"
}
resource "aws_lambda_function" "f" {
  function_name = "f"
  timeout       = 300
}
resource "aws_lambda_event_source_mapping" "m" {
  event_source_arn = aws_sqs_queue.q.arn
  function_name    = aws_lambda_function.f.arn
}
'''
        issues = _run(text)
        self.assertEqual(len(issues), 1)
        self.assertIn("visibility_timeout_seconds(30)", issues[0]["description"])
        self.assertIn("1800", issues[0]["suggestion"])

    def test_ok_when_visibility_sufficient(self):
        text = '''
resource "aws_sqs_queue" "q" {
  visibility_timeout_seconds = 1800
}
resource "aws_lambda_function" "f" {
  timeout = 300
}
resource "aws_lambda_event_source_mapping" "m" {
  event_source_arn = aws_sqs_queue.q.arn
  function_name    = aws_lambda_function.f.arn
}
'''
        self.assertEqual(_run(text), [])


class SecurityGroupTests(unittest.TestCase):
    def test_open_ssh_rule_resource_and_inline(self):
        text = '''
resource "aws_security_group_rule" "ssh" {
  type        = "ingress"
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
}
resource "aws_security_group" "app" {
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
'''
        self.assertEqual(len(_run(text)), 2)

    def test_alb_80_443_open_is_fine(self):
        text = '''
resource "aws_security_group_rule" "http" {
  type        = "ingress"
  from_port   = 80
  to_port     = 80
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
}
resource "aws_security_group_rule" "https" {
  type        = "ingress"
  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
}
resource "aws_security_group_rule" "db" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.app.id
}
'''
        self.assertEqual(_run(text), [])


if __name__ == "__main__":
    unittest.main()
