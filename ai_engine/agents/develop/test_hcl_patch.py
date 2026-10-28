"""hcl_patch 블록 파서·패치 적용 테스트 (LLM 호출 없음)."""
import unittest

from ai_engine.agents.develop.hcl_patch import apply_patch, parse_blocks

MAIN = '''resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = { Name = "main" }   # 인라인 맵 중괄호
}

# 주석 안의 { 는 무시
resource "aws_iam_role_policy" "x" {
  policy = jsonencode({
    Statement = [{ Action = "s3:GetObject", Resource = "${aws_s3_bucket.b.arn}/*" }]
  })
}

data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami"
}

locals {
  tags = { project = "p" }
}

resource "aws_lb" "api" {
  name     = "api"
  internal = false
  tags     = { Name = "${var.name == "x" ? "a}" : "b"}" }
}
'''


class ParseBlocksTests(unittest.TestCase):
    def test_top_level_addresses_in_order(self):
        addrs = [b.address for b in parse_blocks(MAIN)]
        self.assertEqual(addrs, ["resource.aws_vpc.main", "resource.aws_iam_role_policy.x",
                                 "data.aws_ssm_parameter.al2023", "locals", "resource.aws_lb.api"])

    def test_block_text_spans_exact_block(self):
        b = [b for b in parse_blocks(MAIN) if b.address == "resource.aws_lb.api"][0]
        self.assertTrue(MAIN[b.start:b.end].startswith('resource "aws_lb" "api" {'))
        self.assertTrue(MAIN[b.start:b.end].endswith("}"))

    def test_heredoc_braces_ignored(self):
        text = 'resource "aws_iam_policy" "p" {\n  policy = <<EOF\n{ "a": { "b": 1 } }\nEOF\n}\n\noutput "o" {\n  value = 1\n}\n'
        self.assertEqual([b.address for b in parse_blocks(text)], ["resource.aws_iam_policy.p", "output.o"])


class ApplyPatchTests(unittest.TestCase):
    def setUp(self):
        self.files = {"main.tf": MAIN, "variables.tf": 'variable "name" {\n  type = string\n}\n'}

    def test_replace_keeps_other_blocks_byte_identical(self):
        new = 'resource "aws_lb" "api" {\n  name     = "api"\n  internal = true\n}'
        files, applied, skipped = apply_patch(self.files, [
            {"file": "main.tf", "action": "replace", "address": "resource.aws_lb.api", "hcl": new}])
        self.assertEqual(applied, ["replace resource.aws_lb.api"]); self.assertEqual(skipped, [])
        before = MAIN[: MAIN.index('resource "aws_lb" "api"')]
        self.assertTrue(files["main.tf"].startswith(before))     # 앞부분 그대로
        self.assertIn("internal = true", files["main.tf"])
        self.assertNotIn("internal = false", files["main.tf"])

    def test_add_appends_and_delete_removes(self):
        files, applied, _ = apply_patch(self.files, [
            {"file": "variables.tf", "action": "add", "address": "variable.allowed_cidr",
             "hcl": 'variable "allowed_cidr" {\n  type = string\n}'},
            {"file": "main.tf", "action": "delete", "address": "data.aws_ssm_parameter.al2023"},
        ])
        self.assertEqual(applied, ["add variable.allowed_cidr", "delete data.aws_ssm_parameter.al2023"])
        self.assertIn('variable "allowed_cidr"', files["variables.tf"])
        self.assertNotIn("aws_ssm_parameter", files["main.tf"])
        self.assertEqual(len(parse_blocks(files["main.tf"])), 4)

    def test_mismatched_hcl_is_skipped(self):
        files, applied, skipped = apply_patch(self.files, [
            {"file": "main.tf", "action": "replace", "address": "resource.aws_lb.api",
             "hcl": 'resource "aws_lb" "renamed" {\n  name = "x"\n}'}])
        self.assertEqual(applied, []); self.assertEqual(len(skipped), 1)
        self.assertEqual(files["main.tf"], MAIN)

    def test_replace_missing_becomes_add_and_add_existing_becomes_replace(self):
        files, applied, _ = apply_patch(self.files, [
            {"file": "main.tf", "action": "replace", "address": "output.alb_dns",
             "hcl": 'output "alb_dns" {\n  value = aws_lb.api.dns_name\n}'},
            {"file": "main.tf", "action": "add", "address": "resource.aws_vpc.main",
             "hcl": 'resource "aws_vpc" "main" {\n  cidr_block = "10.1.0.0/16"\n}'},
        ])
        self.assertEqual(applied, ["add output.alb_dns", "replace resource.aws_vpc.main"])
        self.assertIn("10.1.0.0/16", files["main.tf"]); self.assertNotIn("10.0.0.0/16", files["main.tf"])


if __name__ == "__main__":
    unittest.main()
