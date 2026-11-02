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


class RdsEngineVersionTests(unittest.TestCase):
    def test_mysql_80_flagged_as_extended_support(self):
        issues = _run('resource "aws_db_instance" "db" {\n  engine = "mysql"\n  engine_version = "8.0"\n  storage_encrypted = true\n}\n')
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["category"], "cost")
        self.assertIn("Extended Support", issues[0]["description"])
        self.assertIn('"8.4"', issues[0]["suggestion"])

    def test_postgres_13_flagged_and_16_ok(self):
        bad = _run('resource "aws_db_instance" "db" {\n  engine = "postgres"\n  engine_version = "13.12"\n  storage_encrypted = true\n}\n')
        ok = _run('resource "aws_db_instance" "db" {\n  engine = "postgres"\n  engine_version = "16"\n  storage_encrypted = true\n}\n')
        self.assertEqual(len(bad), 1); self.assertEqual(ok, [])

    def test_mysql_84_and_no_version_ok(self):
        self.assertEqual(_run('resource "aws_db_instance" "db" {\n  engine = "mysql"\n  engine_version = "8.4"\n  storage_encrypted = true\n}\n'), [])
        self.assertEqual(_run('resource "aws_db_instance" "db" {\n  engine = "mysql"\n  storage_encrypted = true\n}\n'), [])


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


POOL = 'resource "aws_cognito_user_pool" "users" {\n  name = "users"\n}\n'
HTTP_API = 'resource "aws_apigatewayv2_api" "http" {\n  protocol_type = "HTTP"\n}\n'
WS_API = 'resource "aws_apigatewayv2_api" "ws" {\n  protocol_type = "WEBSOCKET"\n}\n'


def _route(name, api, key, authorizer=False):
    extra = '  authorization_type = "JWT"\n  authorizer_id = aws_apigatewayv2_authorizer.jwt.id\n' if authorizer else ""
    return f'resource "aws_apigatewayv2_route" "{name}" {{\n  api_id = aws_apigatewayv2_api.{api}.id\n  route_key = "{key}"\n{extra}}}\n'


class CognitoApiAuthTests(unittest.TestCase):
    def test_http_api_without_authorizer_flagged(self):
        issues = _run(POOL + HTTP_API + _route("default", "http", "$default"))
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["category"], "security")
        self.assertIn("HTTP API", issues[0]["description"])
        self.assertIn("aws_apigatewayv2_authorizer", issues[0]["suggestion"])

    def test_http_api_with_one_protected_route_ok(self):
        text = POOL + HTTP_API + _route("public", "http", "GET /health") + _route("items", "http", "GET /items", authorizer=True)
        self.assertEqual(_run(text), [])

    def test_websocket_connect_without_authorizer_flagged(self):
        text = POOL + WS_API + _route("connect", "ws", "$connect") + _route("send", "ws", "sendMessage")
        issues = _run(text)
        self.assertEqual(len(issues), 1)
        self.assertIn("$connect", issues[0]["description"])

    def test_websocket_connect_with_authorizer_ok(self):
        self.assertEqual(_run(POOL + WS_API + _route("connect", "ws", "$connect", authorizer=True)), [])

    def test_rest_api_without_cognito_method_flagged(self):
        text = POOL + 'resource "aws_api_gateway_rest_api" "rest" {\n  name = "rest"\n}\n' \
            + 'resource "aws_api_gateway_method" "get" {\n  rest_api_id = aws_api_gateway_rest_api.rest.id\n  authorization = "NONE"\n}\n'
        issues = _run(text)
        self.assertEqual(len(issues), 1)
        self.assertIn("REST API", issues[0]["description"])

    def test_rest_api_with_cognito_authorizer_ok(self):
        text = POOL + 'resource "aws_api_gateway_rest_api" "rest" {\n  name = "rest"\n}\n' \
            + 'resource "aws_api_gateway_authorizer" "cog" {\n  rest_api_id = aws_api_gateway_rest_api.rest.id\n  type = "COGNITO_USER_POOLS"\n}\n' \
            + 'resource "aws_api_gateway_method" "get" {\n  rest_api_id = aws_api_gateway_rest_api.rest.id\n  authorization = "COGNITO_USER_POOLS"\n  authorizer_id = aws_api_gateway_authorizer.cog.id\n}\n'
        self.assertEqual(_run(text), [])

    def test_no_cognito_means_no_check(self):
        self.assertEqual(_run(HTTP_API + _route("default", "http", "$default")), [])


class ApigwLambdaPermissionTests(unittest.TestCase):
    def _perm(self, source_arn):
        return ('resource "aws_lambda_permission" "p" {\n  principal = "apigateway.amazonaws.com"\n'
                f'  source_arn = {source_arn}\n}}\n')

    def test_bare_execution_arn_flagged(self):
        issues = _run(self._perm("aws_apigatewayv2_api.ws.execution_arn"))
        self.assertEqual(len(issues), 1)
        self.assertIn("403", issues[0]["description"])

    def test_with_suffix_ok(self):
        self.assertEqual(_run(self._perm('"${aws_apigatewayv2_api.ws.execution_arn}/*/*"')), [])
        self.assertEqual(_run(self._perm('"${aws_apigatewayv2_api.ws.execution_arn}/authorizers/*"')), [])

    def test_other_principal_ignored(self):
        text = 'resource "aws_lambda_permission" "p" {\n  principal = "s3.amazonaws.com"\n  source_arn = aws_s3_bucket.b.arn\n}\n'
        self.assertEqual(_run(text), [])


class WebsocketDefaultStageTests(unittest.TestCase):
    def test_websocket_default_stage_flagged(self):
        text = WS_API + 'resource "aws_apigatewayv2_stage" "s" {\n  api_id = aws_apigatewayv2_api.ws.id\n  name = "$default"\n}\n'
        issues = _run(text)
        self.assertEqual(len(issues), 1)
        self.assertIn("$default", issues[0]["description"])

    def test_http_default_stage_ok(self):
        text = HTTP_API + 'resource "aws_apigatewayv2_stage" "s" {\n  api_id = aws_apigatewayv2_api.http.id\n  name = "$default"\n}\n'
        self.assertEqual(_run(text), [])

    def test_websocket_named_stage_ok(self):
        text = WS_API + 'resource "aws_apigatewayv2_stage" "s" {\n  api_id = aws_apigatewayv2_api.ws.id\n  name = var.environment\n}\n'
        self.assertEqual(_run(text), [])


if __name__ == "__main__":
    unittest.main()
